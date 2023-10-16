def map_number_to_scenario(number):
    if number == 0:
        return "IRTSystemXScenario"
    elif number == 1:
        return "LuSTNanoScenario"
    elif number == 2:
        return "LuSTMiniScenario"
    elif number == 3:
        return "LuSTScenario"
    elif number == 4:
        return "UlmScenario"
    else:
        raise ValueError("Unknow scenario")


def substitute_first_by_param(content, param):
    content = list(content)
    param_plus_space = " = " + str(param) + " \n"
    line_end = ""
    for i in range(len(content)):
        line_end = line_end + "".join(content[i])
        if content[i] == "\n":
            break
    content = "".join(content)
    content = content.replace(line_end, param_plus_space, 1)
    # print(param)
    # print(content)
    return content


def change_single_param(content, name, param):
    content_list = content.split(name)
    substitute = substitute_first_by_param(content_list[1], param)
    new_content = content_list[0] + name
    new_content = new_content + substitute
    return new_content


def change_content_according_to_params(content, app, check, attacker_density):
    new_content = change_single_param(
        content=content, name="checksVersionV1", param=check
    )
    new_content = change_single_param(content=new_content, name="appTypeV1", param=app)
    new_content = change_single_param(
        content=new_content, name="LOCAL_ATTACKER_PROB", param=attacker_density
    )
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
    attacker_density = float(sys.argv[4])
    if attacker_density > 1:
        attacker_density = attacker_density / 100
    if not (attacker_density >= 0 and attacker_density <= 1):
        raise ValueError("Attacker density out of bounds")
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
            new_content = change_content_according_to_params(
                content, app, check, attacker_density
            )
            with open(ini_path, "w") as o_file:
                o_file.write(new_content)
                o_file.close()
