import os
import sys
from ZeusAPI import Client

SERVER = "zeus.microcom.es"
PORT = "4040"
SSL = True

AUTHACCOUNT = os.getenv("ZEUS_AUTHACCOUNT")
PASSWORD = os.getenv("ZEUS_PASSWORD")
AUTHPASS = os.getenv("ZEUS_AUTHPASS")

if not AUTHACCOUNT or not PASSWORD or not AUTHPASS:
    print("Faltan variables de entorno:")
    print("  ZEUS_AUTHACCOUNT")
    print("  ZEUS_PASSWORD")
    print("  ZEUS_AUTHPASS")
    sys.exit(1)

client = Client(SERVER, PORT, SSL, AUTHACCOUNT, PASSWORD, AUTHPASS)

if client.IsCreated() == False:
    print("ERROR creando cliente:", client.GetZeusApiLastError())
    sys.exit(1)

print("OK: autenticación correcta")

stations = client.Stations()
if stations is None:
    print("ERROR obteniendo estaciones:", client.GetZeusApiLastError())
    sys.exit(1)

print(f"Estaciones accesibles: {len(stations)}")
for i, st in enumerate(stations[:20], start=1):
    print(f"{i:02d}. {st}")