def check_config(
        config:dict
):  
    """
    Check the validity of the configuration dictionary based on your logic.
    """
    assert isinstance(config, dict)
    messages = []
    
    # TODO: check validity of arguments
    if not config:
        messages.append("empty config")

    # Print all the messages
    if messages:
        messages = "\n".join(messages)
        print(f"[!] Error in config.json:\n{messages}")
        exit()
    
    return