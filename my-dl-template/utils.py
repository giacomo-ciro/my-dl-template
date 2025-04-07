import sys

def check_config(
        config:dict
):  
    message = ""
    
    # TODO: check validity of arguments

    if message:
        print(f"[!] Error in config.json:\n{message}")
        sys.exit()
    
    return