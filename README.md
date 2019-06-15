# Simpsons Image Classification -

## CNN

### How to run Locally

```bash
git clone https://github.com/ravirane1985/simpsons-classification.git
cd CNN
python simpsons.py
```
### How To Run on Argo cluster

### Setup

```bash
login to Argo
module load keras/2.2.0-py36
mkdir simpsons
cd simpsons
mkdir python
git clone https://github.com/ravirane1985/simpsons-classification.git
pip3 install -r simpsons-classification/requirements.txt -t python
scp -r SampleData <userid>@<serverid>:/scratch/<userid>   // Copy sample data 
cd simpsons-classification/CNN/
sbatch cnn.slurm   // need to follow this directory structure to load dependencies correctly using this slurm script
squeue -u <userid>   // verify the submitted job
/*
  /scratch/<userid> will have log files. *.out will have all print logs
*/
```

### To connect to GitLab repo using a GUI Git tool - GitHub Desktop 

Install GitHub Desktop
In GitLab  -> Settings, generate an Access token.  Save this token.
In GitLab repo, save the URL displayed in the 'Clone with HTTPS' link 
In GitHub Desktop -> Clone repository, use the URL (saved in earlier step) to clone the repo on your local machine.
When prompted for username/password (this may be either during clone operation or first write to repo operation), use GMU email as username and token (saved earlier) as password. 
