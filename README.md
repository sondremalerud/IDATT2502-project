## Demo
![Super Mario Bros. DDQN Demo](demo.gif)

## Requirements

### Python
This project was developed and tested only on Python 3.11, mileage on other versions may vary based on whether dependencies such as Pytorch are available for your Python version

### A C++ compiler
As the nes-py dependency bundles a NES emulator build in C++, a compiler is required (GCC recommended)

### Python packages
All dependencies can be installed by running `pip install -r requirements.txt` from the root of the project

### A copy of Super Mario Bros.
A dumped copy of Super Mario Bros. for the NES is needed for the project to be ran, and should be placed in the root of the project with file name `mario.nes`. The MD5 hash of our game file is `811b027eaf99c2def7b933c5208636de`, yours should match to ensure compatibility.
