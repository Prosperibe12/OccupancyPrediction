import os 
import random
from pipelines import training_pipeline as tp 
from pipelines import deployment_pipeline as dp 
from pipelines import inference_pipeline as ip 
from src.config import Configparams


config = Configparams()
# Get the files from datalake
data_path = [os.path.join(os.path.dirname(__file__), "datalake", file) for file in os.listdir(os.path.join(os.path.dirname(__file__), "datalake"))]

# Select needed columns
cols = [1, 2, 3, 4, 5, 6, 7]

if __name__ == "__main__":

    # execute the training pipeline
    run, acc, df = tp.train_pipeline(
        data_path,
        cols,
        config
    )
    # deployment pipeline
    dp.continuous_deployment_pipeline(acc,run,config)
    # inference pipeline 
    output  = ip.make_inference(random.choice(df), config)
    print("Output", output)