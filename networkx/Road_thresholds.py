input_file = open("./Road_network_sub-dataset.v2")
output_file = "./Road_ranking"

with open(output_file, 'w') as file_object:
    while True:
        line = input_file.readline()
        if not line:
            break
        parts = line.split('	', 8)
        way_id = parts[0]
        limt = int(parts[6])

        if limt == 2:
            speed_class = 130
        elif limt == 3:
            speed_class = 100
        elif limt == 4:
            speed_class = 90
        elif limt == 5:
            speed_class = 70
        elif limt == 6:
            speed_class = 50
        elif limt == 7:
            speed_class = 30
        elif limt == 8:
            speed_class = 11
        else:
            speed_class = limt  # default to original limt if not matched

        file_object.write(f"{way_id},{speed_class}\n")
