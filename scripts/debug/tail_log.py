try:
    with open('mega_training_log.txt', 'rb') as f:
        f.seek(0, 2)
        f_size = f.tell()
        f.seek(max(f_size - 2000, 0))
        lines = f.readlines()
        for line in lines[-5:]:
            print(line.decode('utf-8').strip())
except Exception as e:
    print(f"Error: {e}")
