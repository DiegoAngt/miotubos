"""Pruebas de identidad entre recortes, curación y conteo sin anotaciones inventadas."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box

from common.yolo.counting import (
    Deteccion, exportar_instancias, fusionar, geometria_mascara,
    mascara_local, predecir_instancias, ventanas,
)
from common.yolo.device import seleccionar_dispositivo
from common.yolo.review import compilar_revision
from common.yolo.evaluation import evaluar


class RecortesTest(unittest.TestCase):
    def test_cobertura_incluye_bordes_y_imagenes_pequenas(self):
        for ancho,alto in [(10,12),(512,512),(519,1030)]:
            cobertura=np.zeros((alto,ancho),dtype=bool)
            regiones=ventanas(ancho,alto)
            self.assertEqual(len(regiones),len(set(regiones)))
            for x1,y1,x2,y2 in regiones:
                cobertura[y1:y2,x1:x2]=True
                self.assertLessEqual(x2-x1,512)
                self.assertLessEqual(y2-y1,512)
            self.assertTrue(cobertura.all())

    def test_no_solapamiento_invalido(self):
        with self.assertRaises(ValueError):
            ventanas(512,512,solapamiento=1)

    def test_mascara_con_hueco_y_componentes_no_crea_puentes(self):
        mascara=np.zeros((40,60),dtype=np.uint8)
        mascara[2:20,2:20]=1
        mascara[7:12,7:12]=0
        mascara[25:38,40:58]=1
        g=geometria_mascara(mascara,100,200)
        self.assertEqual(g.geom_type,"MultiPolygon")
        restaurada,bounds=mascara_local(g,200,300)
        x,y,_,_=bounds
        self.assertEqual(restaurada[208-y,108-x],0)
        self.assertEqual(restaurada[230-y,130-x],0)
        self.assertEqual(restaurada[230-y,145-x],1)

    def test_objeto_largo_se_cuenta_una_vez(self):
        entradas=[Deteccion(box(0,20,512,60),.8,0,0),
                  Deteccion(box(256,20,768,60),.9,0,1),
                  Deteccion(box(512,20,1024,60),.7,0,2)]
        salida=fusionar(entradas)
        self.assertEqual(len(salida),1)
        self.assertEqual(salida[0]["geometria"].bounds,(0,20,1024,60))

    def test_cajas_que_se_cruzan_no_fusionan_mascaras_distintas(self):
        a=Polygon([(0,0),(100,100),(95,100),(0,5)])
        b=Polygon([(100,0),(0,100),(0,95),(95,0)])
        self.assertEqual(len(fusionar([Deteccion(a,.9,0,0),Deteccion(b,.9,0,1)])),2)

    def test_no_colapsa_instancias_del_mismo_recorte_por_transitividad(self):
        entradas=[Deteccion(box(0,0,20,20),.9,0,0),
                  Deteccion(box(21,0,41,20),.9,0,0),
                  Deteccion(box(0,0,41,20),.8,0,1)]
        self.assertEqual(len(fusionar(entradas)),2)

    def test_no_fusiona_clases_diferentes(self):
        self.assertEqual(len(fusionar([Deteccion(box(0,0,10,10),.9,0,0),
                                      Deteccion(box(0,0,10,10),.9,1,1)])),2)

    def test_exporta_cero_y_preserva_mascaras_superpuestas(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz=Path(temp)
            imagen=np.zeros((64,64,3),dtype=np.uint8)
            exportar_instancias([],imagen,raiz/"vacio")
            self.assertEqual(json.loads((raiz/"vacio/instancias.json").read_text()),[])
            self.assertEqual(cv2.imread(str(raiz/"vacio/ids.png"),cv2.IMREAD_UNCHANGED).max(),0)
            instancias=[{"id":1,"geometria":box(1,1,20,20),"confianza":.9,"clase":0,"recortes":[0],"toca_borde":True},
                        {"id":2,"geometria":box(10,10,30,30),"confianza":.8,"clase":0,"recortes":[0],"toca_borde":False}]
            exportar_instancias(instancias,imagen,raiz/"dos")
            with np.load(raiz/"dos/mascaras.npz",allow_pickle=False) as datos:
                self.assertEqual(len(datos.files),3)
                self.assertEqual(datos["bbox"].shape,(2,4))
            mapa=cv2.imread(str(raiz/"dos/ids.png"),cv2.IMREAD_UNCHANGED)
            self.assertEqual(mapa[15,15],1)
            self.assertEqual(mapa[25,25],2)

    def test_inferencia_vacia_por_lotes_no_inventa_instancias(self):
        from types import SimpleNamespace
        class Modelo:
            def __init__(self): self.lotes=[]
            def predict(self,fotos,**kwargs):
                self.lotes.append(len(fotos))
                return [SimpleNamespace(boxes=[],masks=None) for _ in fotos]
        modelo=Modelo()
        instancias,stats=predecir_instancias(modelo,np.zeros((600,900,3),dtype=np.uint8),device="cpu",batch=2)
        self.assertEqual(instancias,[])
        self.assertLessEqual(max(modelo.lotes),2)
        self.assertEqual(sum(modelo.lotes),stats["recortes"])

    def test_inferencia_desplaza_y_fusiona_un_objeto_visto_en_dos_recortes(self):
        import torch
        from ultralytics.engine.results import Results
        class Modelo:
            def predict(self,fotos,**kwargs):
                self_test.assertTrue(kwargs["retina_masks"])
                self_test.assertFalse(kwargs["tile"])
                resultados=[]
                for foto,x in zip(fotos,[300,44]):
                    mask=np.zeros(foto.shape[:2],dtype=np.uint8)
                    mask[200:261,x:x+151]=1
                    resultados.append(Results(foto,path="a.png",names={0:"miotubo"},
                        boxes=torch.tensor([[x,200,x+151,261,.9,0]]),
                        masks=torch.from_numpy(mask[None])))
                return resultados
        self_test=self
        salida,stats=predecir_instancias(Modelo(),np.zeros((512,768,3),dtype=np.uint8),device="cpu",batch=2)
        self.assertEqual(stats["detecciones_brutas"],2)
        self.assertEqual(len(salida),1)
        self.assertEqual(salida[0]["geometria"].bounds,(300,200,450,260))
        self.assertFalse(salida[0]["toca_borde"])


class ConteosTest(unittest.TestCase):
    def test_no_evalua_sin_revision_manual(self):
        with self.assertRaises(ValueError):
            evaluar({"a":{"conteo":"3","en_borde":"1"}}, {"a":{"revisado":"no"}})

    def test_error_y_cero_manual(self):
        pred={"a":{"conteo":"3","en_borde":"1"},"b":{"conteo":"2","en_borde":"0"}}
        real={"a":{"revisado":"si","conteo_manual":"5","borde_manual":"2"},
              "b":{"revisado":"si","conteo_manual":"0","borde_manual":"0"}}
        salida=evaluar(pred,real)
        self.assertEqual(salida["mae"],2)
        self.assertEqual(salida["sesgo"],0)
        self.assertEqual(salida["mae_borde"],.5)

    def test_conteos_invalidos_no_se_ignoran(self):
        pred={"a":{"conteo":"3","en_borde":"1"}}
        for valor in ["", "-1", "2.5"]:
            with self.assertRaises(ValueError):
                evaluar(pred,{"a":{"revisado":"si","conteo_manual":valor,"borde_manual":"0"}})


class DispositivoTest(unittest.TestCase):
    @patch("torch.cuda.device_count",return_value=1)
    @patch("torch.cuda.is_available",return_value=True)
    def test_cuda_prioritaria_y_alias(self,*_):
        self.assertEqual(seleccionar_dispositivo(),"0")
        self.assertEqual(seleccionar_dispositivo("cuda:0"),"0")
        with self.assertRaises(ValueError): seleccionar_dispositivo("1")

    @patch("torch.cuda.is_available",return_value=False)
    def test_no_cae_a_cpu_si_se_solicita_nvidia(self,*_):
        with self.assertRaises(ValueError): seleccionar_dispositivo("0")


class RevisionTest(unittest.TestCase):
    def crear_lote(self,raiz):
        (raiz/"images").mkdir()
        (raiz/"labels").mkdir()
        filas=[]
        for i,split in enumerate(["train","val","test"]):
            nombre=f"{i}.png"
            Image.new("RGB",(64,64),(i*70,0,0)).save(raiz/"images"/nombre)
            datos={"imageWidth":64,"imageHeight":64,"shapes":[
                {"label":"miotubo","shape_type":"polygon","group_id":None,
                 "points":[[0,0],[20,0],[20,20],[0,20]]}]}
            (raiz/"labels"/f"{i}.json").write_text(json.dumps(datos))
            filas.append({"imagen":nombre,"split":split,"grupo_biologico":str(i),"anotaciones_completas":"si"})
        return filas

    def guardar(self,raiz,filas):
        with (raiz/"revision.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)

    def test_exporta_portable_sin_rutas_locales(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz=Path(temp);filas=self.crear_lote(raiz);self.guardar(raiz,filas)
            yaml=compilar_revision(raiz)
            self.assertNotIn(str(raiz),yaml.read_text())
            self.assertTrue((yaml.parent/"images/test/2.png").exists())
            self.assertTrue(json.loads((yaml.parent/"protocolo.json").read_text())["anotaciones_completas"])

    def test_rechaza_revision_pendiente_y_fuga_de_grupos(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz=Path(temp);filas=self.crear_lote(raiz)
            for fila in filas: fila["anotaciones_completas"]="no"
            self.guardar(raiz,filas)
            with self.assertRaises(ValueError): compilar_revision(raiz)
            for fila in filas:
                fila["anotaciones_completas"]="si";fila["grupo_biologico"]="mismo"
            self.guardar(raiz,filas)
            with self.assertRaises(ValueError): compilar_revision(raiz)

    def test_rechaza_imagen_duplicada_entre_particiones(self):
        with tempfile.TemporaryDirectory() as temp:
            raiz=Path(temp);filas=self.crear_lote(raiz)
            (raiz/"images/1.png").write_bytes((raiz/"images/0.png").read_bytes())
            self.guardar(raiz,filas)
            with self.assertRaises(ValueError): compilar_revision(raiz)


class EntrenamientoTest(unittest.TestCase):
    def test_cuda_lotes_cache_y_recortes_llegan_a_train_y_val(self):
        from types import SimpleNamespace
        from common.yolo import training as yolo_training
        llamadas=[]
        class Modelo:
            def __init__(self,weights):
                self.weights=weights
            def train(self,**kwargs):
                llamadas.append(("train",kwargs))
                salida=Path(kwargs["project"])/kwargs["name"]
                (salida/"weights").mkdir(parents=True)
                (salida/"weights/best.pt").write_bytes(b"checkpoint de prueba")
                self.trainer=SimpleNamespace(save_dir=salida)
            def val(self,**kwargs):
                llamadas.append(("val",kwargs))
                return SimpleNamespace(results_dict={"metrics/mAP50(M)":.5})
        with tempfile.TemporaryDirectory() as temp:
            with patch("ultralytics.YOLO",Modelo), patch.object(yolo_training,"RESULTADOS",Path(temp)), patch.object(yolo_training,"seleccionar_dispositivo",return_value="0"):
                yolo_training.entrenar(Path(temp)/"dataset.yaml","yolo",50,"0",batch=8,workers=4,
                                       cache="ram",tile=512,imgsz=512)
            train,val=llamadas[0][1],llamadas[1][1]
            self.assertTrue(train["amp"])
            self.assertEqual(train["cache"],"ram")
            self.assertEqual(train["epochs"],50)
            for kwargs in (train,val):
                self.assertEqual(kwargs["device"],"0")
                self.assertEqual(kwargs["batch"],8)
                self.assertEqual(kwargs["workers"],4)
                self.assertEqual(kwargs["tile"],512)
                self.assertEqual(kwargs["imgsz"],512)


if __name__ == "__main__":
    unittest.main()
