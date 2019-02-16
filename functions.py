def read_counter(filepath):
    file = open(filepath, "r")
    curr_id = int(file.readline().strip())
    file.close()

    file = open(filepath, "w")
    file.write(str(curr_id + 1))
    file.close()

    return curr_id
