# RPG
First year project, classical RPG made with CSFML library.

## Features

* In-game save system
* Turn-based combat with enemy A.I.
* Movement management on a map
* Animated effects (spells)
* PNJ
* Inventory management and loot
* Settings

## Build with linux (ubuntu)

Realised with CSFML 2.5

### 1. Install the dependencies:

	sudo apt-get install build-essential libpthread-stubs0-dev libgl1-mesa-dev libx11-dev libxrandr-dev libfreetype6-dev libglew-dev libjpeg8-dev libsndfile1-dev libopenal-dev
	sudo apt-get install libsfml-dev

### 2. Download the CSFML source code and extract it:
	https://www.sfml-dev.org/download/csfml/

### 3. Build and install CSFML:

    mkdir build
	cd build
	cmake ..
	make
	sudo make install

### 4. Build my_rpg from the repo:
	make
	./my_rpg
