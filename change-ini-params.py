def map_number_to_scenario(number):
    if number == 1:
        return "IRTSystemXScenario"
    elif number == 2:
        return "LuSTNanoScenario"
    elif number == 3:
        return "LuSTMiniScenario"
    elif number == 4:
        return "LuSTScenario"
    elif number == 5:
        return "UlmScenario"
    else:
        raise ValueError("Unknow scenario")

def substitute_first_by_int(content, number):
    content = list(content)
    for i in range(len(content)):
        if content[i].isdigit():
            number_char = str(number)[0]
            content[i] = number_char
            break
    return "".join(content)

def change_content_according_to_params(content,app,check):
    content_list = content.split("checksVersionV1")
    substituted_check = substitute_first_by_int(content_list[1],check)
    content_list = [content_list[0], substituted_check]
    new_content = content_list[0] + "checksVersionV1" + content_list[1]
    content_list = new_content.split("appTypeV1")
    substituted_app = substitute_first_by_int(content_list[1],app)
    content_list = [content_list[0], substituted_app]
    new_content = content_list[0] + "appTypeV1" + content_list[1]
    # print(new_content)
    return new_content

if __name__ == "__main__":
    import os
    import sys

    network_path = "/F2MD/veins-f2md/f2md-networks/"
    scenario_int = int(sys.argv[1])
    app = int(sys.argv[2])
    if not (app <= 5 and app >= 0):
        raise ValueError("App type out of bounds")
    check = int(sys.argv[3])
    if not (check <= 2 and check >= 0):
        raise ValueError("Check type out of bounds")
    scenario_string = map_number_to_scenario(scenario_int)
    scenario_path = network_path + scenario_string
    if not (os.path.exists(scenario_path) and os.path.isdir(scenario_path)):
        raise ValueError("Path to scenario not right")
    else:
        ini_path = scenario_path + "/omnetpp.ini"
        if not (os.path.exists(scenario_path)):
            raise ValueError("No omnetpp.ini in directory")
        else:
            with open(ini_path, "r") as o_file:
                content = o_file.read()
                o_file.close()
            new_content = change_content_according_to_params(content,app,check)
            with open(ini_path, "w") as o_file:
                o_file.write(new_content)
                o_file.close()