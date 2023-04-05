import common.config as config
import torch
import os
import re


class ModuleSaveLoad:

    def __init__(self):
        # Keep the module
        self.module_list = []

    def save_module(self, main_q_network,  episode=0):
        module_path = config.PATH_MODULE_BACKUP.format(str(episode))
        torch.save(main_q_network.state_dict(), module_path)

        # Save in list
        if len(self.module_list) >= config.BACKUP_AMOUNT:
            # remove the oldest
            module_to_delete = self.module_list.pop(0)
            if os.path.exists(module_to_delete):
                os.remove(module_to_delete)

        self.module_list.append(module_path)

        # Save in file
        if os.path.exists(config.PATH_MODULE_LIST):
            os.remove(config.PATH_MODULE_LIST)
        with open(config.PATH_MODULE_LIST, 'a+') as file:
            for each_path in self.module_list:
                file.writelines(each_path+'\n')

    # Get the saved module
    def load_module(self):
        if os.path.exists(config.PATH_MODULE_LIST):
            with open(config.PATH_MODULE_LIST, 'r') as file:
                lines = file.readlines()
                if len(lines) >= 1:

                    self.module_list = list(map(lambda x: x[0:-1], lines))

                    while len(self.module_list) > config.BACKUP_AMOUNT:
                        self.module_list.pop(0)

                    # Return the last item
                    targe_path = self.module_list[-1]
                    if os.path.exists(targe_path):
                        matches = re.findall(r'\d+', targe_path)
                        starting_episode = int(matches[0])
                        return torch.load(targe_path), starting_episode
                    else:
                        return None, None
                else:
                    return None, None
        else:
            return None, None

