#!/bin/bash
# Script de solución de problemas para Drone OSINT GeoSpy

# Colores para formatear la salida
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Solución de problemas - Drone OSINT GeoSpy  ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Verificar si hay un archivo .env
if [ -f .env ]; then
    echo -e "${GREEN}✓ El archivo .env existe${NC}"
    # Verificar si contiene la clave API de OpenAI
    if grep -q "OPENAI_API_KEY" .env && ! grep -q "OPENAI_API_KEY=sk-xxxxxx" .env; then
        echo -e "${GREEN}✓ OPENAI_API_KEY está configurada en .env${NC}"
    else
        echo -e "${RED}✗ OPENAI_API_KEY no está configurada correctamente en .env${NC}"
        echo -e "${YELLOW}  Solución: Edite el archivo .env y añada su clave API de OpenAI.${NC}"
    fi
else
    echo -e "${RED}✗ No se encontró el archivo .env${NC}"
    echo -e "${YELLOW}  Solución: Copie .env.example a .env y edite el archivo.${NC}"
    if [ -f .env.example ]; then
        echo -e "${YELLOW}  Comando: cp .env.example .env${NC}"
    else
        echo -e "${RED}  .env.example tampoco existe. Deberá crear un archivo .env manualmente.${NC}"
    fi
fi

echo ""
echo -e "${BLUE}== Verificando estado de Docker ==${NC}"

# Comprobar si Docker está instalado
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker está instalado${NC}"
    # Comprobar si Docker está en ejecución
    if docker info &> /dev/null; then
        echo -e "${GREEN}✓ Docker está en ejecución${NC}"
    else
        echo -e "${RED}✗ Docker no está en ejecución${NC}"
        echo -e "${YELLOW}  Solución: Inicie el servicio Docker.${NC}"
    fi
else
    echo -e "${RED}✗ Docker no está instalado${NC}"
    echo -e "${YELLOW}  Solución: Instale Docker siguiendo las instrucciones en https://docs.docker.com/get-docker/${NC}"
fi

# Comprobar si docker-compose está instalado
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓ Docker Compose está instalado${NC}"
else
    echo -e "${RED}✗ Docker Compose no está instalado${NC}"
    echo -e "${YELLOW}  Solución: Instale Docker Compose siguiendo las instrucciones en https://docs.docker.com/compose/install/${NC}"
fi

echo ""
echo -e "${BLUE}== Verificando archivos de configuración ==${NC}"

# Comprobar docker-compose.yml
if [ -f docker-compose.yml ]; then
    echo -e "${GREEN}✓ docker-compose.yml existe${NC}"
else
    echo -e "${RED}✗ docker-compose.yml no existe${NC}"
    echo -e "${YELLOW}  Solución: Verifique que está en el directorio correcto.${NC}"
fi

# Comprobar si hay contenedores ya en ejecución
if command -v docker &> /dev/null && docker info &> /dev/null; then
    running_containers=$(docker ps --filter "name=drone-osint-geospy" --format "{{.Names}}")
    if [ -n "$running_containers" ]; then
        echo -e "${GREEN}✓ Contenedores en ejecución:${NC}"
        echo "$running_containers"
    else
        echo -e "${YELLOW}! No hay contenedores de Drone OSINT GeoSpy en ejecución${NC}"
    fi
fi

echo ""
echo -e "${BLUE}== Verificando dependencias de Python ==${NC}"

# Comprobar si Python está instalado
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo -e "${GREEN}✓ $python_version está instalado${NC}"
else
    echo -e "${RED}✗ Python 3 no está instalado${NC}"
    echo -e "${YELLOW}  Solución: Instale Python 3.8 o superior desde https://www.python.org/downloads/${NC}"
fi

# Comprobar si pip está instalado
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓ pip está instalado${NC}"
else
    echo -e "${RED}✗ pip no está instalado${NC}"
    echo -e "${YELLOW}  Solución: Instale pip para Python 3.${NC}"
fi

# Verificar si requirements.txt existe y contiene las dependencias necesarias
if [ -f requirements.txt ]; then
    echo -e "${GREEN}✓ requirements.txt existe${NC}"
    
    # Verificar dependencias críticas
    critical_deps=("fastapi" "openai" "streamlit" "python-dotenv")
    for dep in "${critical_deps[@]}"; do
        if grep -q "$dep" requirements.txt; then
            echo -e "${GREEN}✓ $dep está en requirements.txt${NC}"
        else
            echo -e "${RED}✗ $dep no está en requirements.txt${NC}"
        fi
    done
else
    echo -e "${RED}✗ requirements.txt no existe${NC}"
    echo -e "${YELLOW}  Solución: Verifique que está en el directorio correcto.${NC}"
fi

echo ""
echo -e "${BLUE}== Pasos recomendados para solucionar problemas ==${NC}"
echo -e "${YELLOW}1. Verifique que .env esté configurado correctamente con su OPENAI_API_KEY${NC}"
echo -e "${YELLOW}2. Para problemas de conexión entre servicios:${NC}"
echo -e "   - Detenga los contenedores: ${BLUE}docker-compose down${NC}"
echo -e "   - Elimine imágenes antiguas: ${BLUE}docker-compose down --rmi all${NC}"
echo -e "   - Reconstruya los contenedores: ${BLUE}docker-compose up --build${NC}"
echo -e "${YELLOW}3. Para problemas persistentes, puede ejecutar la herramienta de depuración:${NC}"
echo -e "   ${BLUE}docker-compose run --rm backend debug${NC}"
echo -e "${YELLOW}4. Para verificar la inicialización del backend:${NC}"
echo -e "   ${BLUE}docker-compose run --rm backend test-backend${NC}"
echo -e "${YELLOW}5. Si los servicios no se comunican, puede ser un problema de red de Docker:${NC}"
echo -e "   ${BLUE}docker network prune${NC}"
echo -e "   Y luego vuelva a iniciar los servicios.${NC}"

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}     Fin de la verificación de problemas     ${NC}"
echo -e "${BLUE}============================================${NC}"

exit 0 