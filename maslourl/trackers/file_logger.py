import os


class FileLogger():

    def __init__(self, file_name='progress.log'):
        self.file_name = file_name
        self.clean_progress_file()

    def log(self, episode, steps, reward, average_reward, epsilon):
        f = open(self.file_name, 'a+')
        f.write(f"{episode};{steps};{reward};{average_reward};{epsilon}\n")
        f.close()

    def clean_progress_file(self):
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        f = open(self.file_name, 'a+')
        f.write("episode;steps;reward;average;epsilon\n")
        f.close()
