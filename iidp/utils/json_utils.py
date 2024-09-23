import json


def read_json(json_file_path):
    try:
        with open(json_file_path, 'r') as jf:
            json_data = json.load(jf)
    except IOError as e:
        print("[json_utils][read_json] I/O error({0}): {1} - {2}".format(e.errno, e.strerror, json_file_path))
        exit(1)
    return json_data


def write_json(json_file_path, data):
    try:
        with open(json_file_path, 'w') as jf:
            json_str = json.dumps(data)
            jf.write(json_str)
    except IOError as e:
        print("[json_utils][write_json] I/O error({0}): {1} - file path: {2} data: {3}".format(e.errno, e.strerror, json_file_path, data))
        exit(1)
