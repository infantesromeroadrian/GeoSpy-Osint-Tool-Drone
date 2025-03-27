#!/usr/bin/env python3
"""
Script para probar la inicialización del cliente OpenAI
"""

import os
import sys
from dotenv import load_dotenv
import logging
import json
import inspect

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_openai")

# Cargar variables de entorno
load_dotenv()

def test_openai_import():
    """Prueba la importación del módulo OpenAI"""
    try:
        import openai
        logger.info(f"OpenAI versión: {openai.__version__}")
        logger.info(f"OpenAI path: {inspect.getfile(openai)}")
        return True
    except ImportError as e:
        logger.error(f"Error al importar OpenAI: {e}")
        return False

def test_openai_client():
    """Prueba la inicialización del cliente OpenAI"""
    try:
        from openai import OpenAI
        
        # Obtener API key
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.error("API key no encontrada. Configure OPENAI_API_KEY en .env")
            return False
            
        logger.info(f"Inicializando cliente con API key: {api_key[:4]}...{api_key[-4:]}")
        logger.info(f"Longitud de API key: {len(api_key)}")
        
        # Intenta diferentes métodos de inicialización
        methods = []
        
        # Método 1: Estándar
        try:
            client = OpenAI(api_key=api_key)
            methods.append({"method": "standard", "success": True})
            logger.info("✅ Método estándar funcionó")
        except Exception as e:
            methods.append({"method": "standard", "success": False, "error": str(e)})
            logger.error(f"❌ Método estándar falló: {e}")
            
        # Método 2: Sin API key (esperando que use la variable de entorno)
        try:
            client = OpenAI()
            methods.append({"method": "env_var", "success": True})
            logger.info("✅ Método con variable de entorno funcionó")
        except Exception as e:
            methods.append({"method": "env_var", "success": False, "error": str(e)})
            logger.error(f"❌ Método con variable de entorno falló: {e}")
            
        # Método 3: Alternativo
        try:
            client = OpenAI()
            client.api_key = api_key
            methods.append({"method": "two_step", "success": True})
            logger.info("✅ Método en dos pasos funcionó")
        except Exception as e:
            methods.append({"method": "two_step", "success": False, "error": str(e)})
            logger.error(f"❌ Método en dos pasos falló: {e}")
            
        # Verificar si algún método funcionó
        if any(m["success"] for m in methods):
            logger.info("🎉 Al menos un método de inicialización funcionó")
            return True
        else:
            logger.error("❌ Todos los métodos de inicialización fallaron")
            return False
    except Exception as e:
        logger.error(f"Error general al inicializar cliente OpenAI: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vision_llm():
    """Prueba la clase VisionLLM"""
    try:
        # Añadir directorio actual al path
        sys.path.append(os.getcwd())
        
        # Intentar importar tanto con 'src.' como sin él
        try:
            from src.models.vision_llm import VisionLLM
            logger.info("VisionLLM importado desde src.models.vision_llm")
            module_path = "src.models.vision_llm"
        except ImportError:
            try:
                from models.vision_llm import VisionLLM
                logger.info("VisionLLM importado desde models.vision_llm")
                module_path = "models.vision_llm"
            except ImportError as e:
                logger.error(f"No se pudo importar VisionLLM: {e}")
                return False
        
        # Obtener API key
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.error("API key no encontrada. Configure OPENAI_API_KEY en .env")
            return False
            
        # Inicializar VisionLLM
        vision_llm = VisionLLM(api_key=api_key)
        if vision_llm.client:
            logger.info("✅ VisionLLM inicializado correctamente")
            return True
        else:
            logger.error("❌ VisionLLM inicializado pero cliente es None")
            return False
    except Exception as e:
        logger.error(f"Error al probar VisionLLM: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_environment():
    """Imprime información del entorno para ayudar en el diagnóstico"""
    logger.info("==== VARIABLES DE ENTORNO ====")
    for key, value in os.environ.items():
        if key in ["OPENAI_API_KEY", "API_KEY"]:
            logger.info(f"{key}: {value[:4]}...{value[-4:]} (length: {len(value)})")
        elif "API" in key or "SECRET" in key or "KEY" in key:
            logger.info(f"{key}: [REDACTED FOR SECURITY]")
        else:
            logger.info(f"{key}: {value}")
            
    logger.info("==== INFORMACIÓN DEL SISTEMA ====")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    # Listar paquetes instalados
    try:
        import pkg_resources
        installed_packages = [(d.project_name, d.version) for d in pkg_resources.working_set]
        logger.info(f"Installed packages: {json.dumps(dict(installed_packages), indent=2)}")
    except Exception as e:
        logger.error(f"Error listing packages: {e}")

if __name__ == "__main__":
    print("\n===== DIAGNÓSTICO DE OPENAI =====")
    
    print_environment()
    
    if test_openai_import():
        print("✅ Importación de OpenAI: OK")
    else:
        print("❌ Importación de OpenAI: ERROR")
        
    if test_openai_client():
        print("✅ Cliente OpenAI: OK")
    else:
        print("❌ Cliente OpenAI: ERROR")
        
    if test_vision_llm():
        print("✅ VisionLLM: OK")
    else:
        print("❌ VisionLLM: ERROR")
    
    print("===== FIN DE DIAGNÓSTICO =====\n") 