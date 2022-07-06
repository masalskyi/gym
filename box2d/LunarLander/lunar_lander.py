from maslourl.models.maslou_rl import MaslouRLModel2QDiscrete
from maslourl.models.utils import masked_rmse
import uuid
import shutil
import tensorflow as tf


class LunarLander2QModel(MaslouRLModel2QDiscrete):
    def __get_params_for_episode(self):
        return {"fraction_finished": 0}

    def __copy_model(self, model):
        backup_file = 'backup_' + str(uuid.uuid4())
        model.save(backup_file)
        new_model = tf.keras.models.load_model(backup_file, custom_objects={'masked_rmse': masked_rmse(0.0)})
        shutil.rmtree(backup_file)
        return new_model

    # returns action that was done, new state, reward, done or not, info
    def step(self, state, target_model, epsilon, params):
        raise NotImplementedError("step must be implemented in child")
