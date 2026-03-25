PRE_LOADED_PATH = "../assets"

# get system name
import platform
system_name = platform.system()
if system_name == "Windows":
    system_name = "windows"
elif system_name == "Linux":
    system_name = "linux"
import os
def get_local_url(url):
    if url.find(system_name) == -1:
        other_name = "linux" if system_name == "windows" else "windows"
        url = url.replace(other_name, system_name)
    url = url.replace("private", "public")
    filedir = url[url.find(".com/") + 5:]
    if os.path.exists(os.path.join(PRE_LOADED_PATH, filedir)):
        return "file://" + str(os.path.join(PRE_LOADED_PATH, filedir))
    else:
        return url

if __name__ == "__main__":
    print(get_local_url("https://tdw-public.s3.amazonaws.com/scenes/windows/2020.3/floorplan_1a"))