import os
import glob

def find_next_path_id(path, name):
  max_run_id = 0

  for path in glob.glob("{}/{}_[0-9]*".format(path, name)):
    file_name = path.split(os.sep)[-1]
    ext = file_name.split("_")[-1]
    if name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
      max_run_id = int(ext)
  return max_run_id

def createNextFileName(tensorboard_log_dir, suffix):
    id = find_next_path_id(tensorboard_log_dir, suffix) + 1
    log_dir = tensorboard_log_dir + "/" + suffix + "_"+ str(id)
    print("Log created at: ", log_dir)
    return  log_dir, id