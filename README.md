# active_learning
Here we are exploring active learning as applied to NV ODMR temeprature sensor. The goal is to create an active learning framework where model uncertainty guides the next temperature measurment in sensor calibration. 
In this project we will rely on biL model and it's derviatives

## Folder structure
Here we are concerned only with ODMR spectra

The folder structure is as following:
- adaptive_sampling: notebooks exploring adative sampling to speed up the probablistic model
- batch_results: train probablistic model on batches of data
- nv_diamond_on_pcb: exploring biL model applied to nv microparticle afixed to pcb antenna
- sensor1_cycle1: notebooks exploring short legged temp range cycle
- sensor1_cycle2: notebooks exploring wider temp range cycle
- sensor1_cycle3: notebook exploring cycle three with wider temp steps 
