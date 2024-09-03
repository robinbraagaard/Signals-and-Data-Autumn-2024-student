# Signals-And-Data-Autumn-2024
This repository contains the exercises for each week of the 02462 Signals and Data course at DTU for autumn 2024. 

# Set-up
To use this repository most effectively, we recommend you set up a virtual environment and use our requirements.txt file to install all the necessary libraries:

0. (Optional): Clone this repository, open a terminal a write: \
  ```git clone https://github.com/02462-Signals-and-Data/Signals-and-Data-Autumn-2024-student.git```

1. Create a new conda environment, open your terminal (or specific conda terminal): \
 ```conda create --name your_env_name```

2. Navigate to the folder with the requirements file and install them all using pip: \
```pip install -r requirements.txt``` 

1. Before running your code (in a notebook), be sure to:  
   1. Select your new environment as your Python interpreter (if using VSCode)
   2. Activate your environment by inputting ```conda acitvate your_env_name``` and open a jupyter notebook with ```jupyter notebook```

*Do remember to ask for help from your TAs, a chatbot or Google if you run into any problems performing the above steps.*

# Pulling updates

During the course, we will both upload new material, and update existing material. To access these new updates, simply navigate to your project repository, open a terminal and write:

```git pull --rebase --autostash```

You're free to use your own methods of pulling and such, but using this method automatically stashes and reapplies your local changes. Meaning:

1. Git doesn't bitch about you not having commited or stashed your changes
2. Git does not overwrite your precious changes to existing weeks (in case we change something in the previous weeks to fix and error, for example)
3. **HOWEVER** in the case of 2. it *might* lead to merge conflicts and the like, as your local changes are applied willy-nilly on the remote changes.

