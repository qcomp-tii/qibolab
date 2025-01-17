import psutil, socket
import Pyro4
import Pyro4.naming
from qick.qick_asm import QickConfig
from qick.qick import QickSoc

ns_host="localhost"
ns_port=8888
bitfile='/home/xilinx/jupyter_notebooks/qick/TII/bitstreams/216_tProc_v25.bit'
peripherals_drivers = '/home/xilinx/jupyter_notebooks/qick/TII/drivers'
iface='eth0'

### INITIALIZE PYRO ###
Pyro4.config.REQUIRE_EXPOSE = False
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED=set(['pickle'])
Pyro4.config.PICKLE_PROTOCOL_VERSION=4

print("looking for nameserver . . .")
ns = Pyro4.locateNS(host=ns_host, port=ns_port)
print("found nameserver")

# if we have multiple network interfaces, we want to register the daemon using the IP address that faces the nameserver
host = Pyro4.socketutil.getInterfaceAddress(ns._pyroUri.host)
# if the nameserver is running on the QICK, the above will usually return the loopback address - not useful
if host=="127.0.0.1":
    # if the eth0 interface has an IPv4 address, use that
    # otherwise use the address of any interface starting with "eth0" (e.g. eth0:1, which is typically a static IP)
    # unless you have an unusual network config (e.g. VPN), this is the interface clients will want to connect to
    for name, addrs in psutil.net_if_addrs().items():
        addrs_v4 = [addr.address for addr in addrs if addr.family==socket.AddressFamily.AF_INET]
        if len(addrs_v4)==1:
            if name.startswith(iface):
                host = addrs_v4[0]
            if name==iface:
                break
daemon = Pyro4.Daemon(host=host)

### INITIALIZE QICK ###
soc = QickSoc(bitfile)
soccfg = QickConfig(soc.get_cfg())
with open(bitfile + ".json", "w") as f:
    f.write(soc.dump_cfg())

# 2250 μA to 40500 μA
dac_2280 = soc.usp_rf_data_converter_0.dac_tiles[0].blocks[0]
dac_2280.SetDACVOP(30000)
dac_2281 = soc.usp_rf_data_converter_0.dac_tiles[0].blocks[1]
dac_2281.SetDACVOP(30000)
dac_2282 = soc.usp_rf_data_converter_0.dac_tiles[0].blocks[2]
dac_2282.SetDACVOP(30000)
dac_2283 = soc.usp_rf_data_converter_0.dac_tiles[0].blocks[3]
dac_2283.SetDACVOP(30000)
dac_2290 = soc.usp_rf_data_converter_0.dac_tiles[1].blocks[0]
dac_2290.SetDACVOP(30000)
dac_2292 = soc.usp_rf_data_converter_0.dac_tiles[1].blocks[2]
dac_2292.SetDACVOP(5000)
dac_2230 = soc.usp_rf_data_converter_0.dac_tiles[2].blocks[0]
dac_2230.SetDACVOP(30000)
dac_2231 = soc.usp_rf_data_converter_0.dac_tiles[2].blocks[1]
dac_2231.SetDACVOP(30000)
dac_2232 = soc.usp_rf_data_converter_0.dac_tiles[2].blocks[2]
dac_2232.SetDACVOP(30000)

soc.usp_rf_data_converter_0.mts_dac_config.RefTile = 2
soc.usp_rf_data_converter_0.mts_dac_config.Tiles = 0b0011
soc.usp_rf_data_converter_0.mts_dac_config.SysRef_Enable = 1
soc.usp_rf_data_converter_0.mts_dac_config.Target_Latency = -1
soc.usp_rf_data_converter_0.mts_dac()
    
### INITIALIZE TIDAC80508 ###
import sys
sys.path.append(peripherals_drivers)
from TIDAC80508 import TIDAC80508
tidac = TIDAC80508()

### INITIALIZE LIBGPIO_CONTROL ###
import ctypes
import time
def logmsg(msg):
    print(f"{time.time()} - {msg}")
class libgpio_control():
    def __init__(self):
        self.lib = ctypes.CDLL('/home/xilinx/jupyter_notebooks/qick/TII/drivers/lib_gpio_control_2.so')
    def initialize_gpio(self):
        logmsg("before initialize_gpio")
        self.lib.initialize_gpio()
        logmsg("after initialize_gpio")
    def set_gpio_low(self):
        logmsg("before set_gpio_low")
        self.lib.set_gpio_low()
        logmsg("after set_gpio_low")
    def set_gpio_high(self):
        logmsg("before set_gpio_high")
        self.lib.set_gpio_high()
        logmsg("after set_gpio_high")
    def cleanup_gpio(self):
        logmsg("before cleanup_gpio")
        self.lib.cleanup_gpio()
        logmsg("after cleanup_gpio")
          
        
print("All objects initialized")

# register the QickSoc in the daemon (so the daemon exposes the QickSoc over Pyro4)
# and in the nameserver (so the client can find the QickSoc)
ns.register("soc", daemon.register(soc))
ns.register("tidac", daemon.register(tidac))
ns.register("libgpio_control", daemon.register(libgpio_control))

print("All objects registered")

# register in the daemon all the objects we expose as properties of the QickSoc
# we don't register them in the nameserver, since they are only meant to be accessed through the QickSoc proxy
# https://pyro4.readthedocs.io/en/stable/servercode.html#autoproxying
# https://github.com/irmen/Pyro4/blob/master/examples/autoproxy/server.py

for obj in soc.autoproxy:
    daemon.register(obj)
    print("registered member "+str(obj))

print("starting daemon")
daemon.requestLoop() # this will run forever until interrupted