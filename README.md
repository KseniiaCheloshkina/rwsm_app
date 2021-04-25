This image contains RWSM model (Ranking Weibull Survival Model). 
Using it you can do the following:
- Try a DEMO:
    - docker run --name rwsm_model -p 5000:5000 rwsm:0.1            # start container
    - curl -X GET http://127.0.0.1:5000/demo      # run demo model training and evaluation on METABRIC dataset
- Change parameters for model training still using default neural network head and dataset:
    - docker run --name rwsm_model -p 5000:5000 rwsm:0.1
    - curl -X POST http://127.0.0.1:5000/train -d '{"model_type": "contrastive"}'
- Use custom dataset / config / neural network head: 
  - mkdir docker_vol # create a folder on host
  - docker run --name rwsm_model -p 5000:5000 -v <WORKDIR_ON_HOST>/docker_vol/:/app/docker_vol/ rwsm:0.1 # start container with volume
  - cp training and testing data (train_data.pkl and test_data.pkl) to `docker_vol` folder as well as config (docker_vol/config_contrastive.json)
  - docker cp custom_nn_script.py rwsm_model:/app/ # cp file with custom neural network head
  - docker exec -it rwsm_model /bin/bash # get into container
  - cat custom_nn_script.py >> custom_models.py # add custom function (NN head) to models file
  - curl -X POST http://127.0.0.1:5000/train -d '{"save_path": "docker_vol/", "model_type": "contrastive", "config_path": "docker_vol/config_contrastive.json", "train_data_path": "docker_vol/train_data.pkl", "val_data_path": "docker_vol/test_data.pkl", "custom_bottom_function_name": "metabric_custom_network"}' 