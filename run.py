#!/usr/bin/env python3
"""
Script para iniciar Drone OSINT GeoSpy
-------------------------------------
Este script facilita la ejecución del sistema tanto en modo Docker como local.
"""

import os
import sys
import argparse
import subprocess
import platform
import time

def check_requirements():
    """Comprobar si están instalados los requisitos necesarios"""
    try:
        # Comprobar Python
        if sys.version_info.major < 3 or (sys.version_info.major == 3 and sys.version_info.minor < 8):
            print("ERROR: Se requiere Python 3.8 o superior.")
            return False
        
        # Comprobar Docker y Docker Compose si usamos esos modos
        if os.system("docker --version > nul 2>&1" if platform.system() == "Windows" else "docker --version > /dev/null 2>&1") != 0:
            print("ADVERTENCIA: Docker no está instalado o no está en el PATH. No se podrá usar el modo Docker.")
            has_docker = False
        else:
            has_docker = True
            
        return True
    except Exception as e:
        print(f"Error al comprobar requisitos: {e}")
        return False

def setup_environment():
    """Configurar el entorno de ejecución"""
    # Copiar .env.example a .env si no existe
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("Creando archivo .env a partir de .env.example...")
            with open(".env.example", "r") as src:
                with open(".env", "w") as dst:
                    dst.write(src.read())
            print("Archivo .env creado. IMPORTANTE: Edite el archivo y configure su API key de OpenAI.")
        else:
            print("ADVERTENCIA: No se encontró .env.example. Cree un archivo .env manualmente.")

def run_docker(mode, development=True, map_volume=True):
    """Ejecutar el sistema en modo Docker"""
    compose_file = "docker-compose.yml" if development else "docker-compose.prod.yml"
    
    if not os.path.exists(compose_file):
        print(f"ERROR: No se encontró el archivo {compose_file}")
        return False
    
    # Verificar si hay contenedores existentes
    try:
        print("Comprobando contenedores existentes...")
        subprocess.run(["docker-compose", "-f", compose_file, "ps", "-q"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("ERROR: No se pudo ejecutar docker-compose. Asegúrese de que está instalado y en el PATH.")
        return False
    
    # Detener contenedores existentes si es necesario
    if mode == "restart":
        print("Deteniendo contenedores existentes...")
        subprocess.run(["docker-compose", "-f", compose_file, "down"], check=True)
    
    # Construir y arrancar
    print(f"Iniciando servicios con {compose_file}...")
    volume_arg = [] if map_volume else ["--no-deps"]
    command = ["docker-compose", "-f", compose_file, "up", "--build"]
    command.extend(volume_arg)
    
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR al iniciar los servicios: {e}")
        return False
    except KeyboardInterrupt:
        print("\nDeteniendo servicios...")
        subprocess.run(["docker-compose", "-f", compose_file, "down"], check=True)
        return True

def run_local(frontend_only=False):
    """Ejecutar el sistema en modo local"""
    try:
        # Verificar instalación de dependencias
        print("Comprobando dependencias...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        if not frontend_only:
            # Iniciar backend
            print("Iniciando backend...")
            backend_process = subprocess.Popen([sys.executable, "src/main.py"])
            time.sleep(2)  # Dar tiempo al backend para iniciar
        
        # Iniciar frontend
        print("Iniciando frontend...")
        env = os.environ.copy()
        if not "API_URL" in env:
            env["API_URL"] = "http://localhost:8000"
        frontend_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/frontend/app.py"], env=env)
        
        # Esperar a que el usuario termine
        try:
            if not frontend_only:
                frontend_process.wait()
                backend_process.terminate()
            else:
                frontend_process.wait()
        except KeyboardInterrupt:
            print("\nDeteniendo servicios...")
            if not frontend_only:
                backend_process.terminate()
            frontend_process.terminate()
        
        return True
    except Exception as e:
        print(f"ERROR al iniciar el sistema: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Drone OSINT GeoSpy - Herramienta de inicialización")
    parser.add_argument("--mode", choices=["docker", "local", "frontend", "restart", "production"], 
                      default="docker", help="Modo de ejecución")
    parser.add_argument("--no-volume", action="store_true", 
                      help="No mapear volúmenes en modo Docker")
    
    args = parser.parse_args()
    
    if not check_requirements():
        sys.exit(1)
    
    setup_environment()
    
    if args.mode in ["docker", "restart", "production"]:
        success = run_docker(args.mode, development=(args.mode != "production"), map_volume=not args.no_volume)
    elif args.mode == "local":
        success = run_local(frontend_only=False)
    elif args.mode == "frontend":
        success = run_local(frontend_only=True)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 