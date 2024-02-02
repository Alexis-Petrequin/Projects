# Sokoban
First year project, a sokoban-like game in terminal that uses .txt files to generate a game map.

## USAGE
        ./my_sokoban 'name of the map'.txt
        
### DESCRIPTION
        'name of the map'.txt :  file represents the warehouse map, ‘#’ are the walls,
             ‘P’ is the player,‘X’ are the boxes and ‘O’ are storage locations.
             

## How to build

### 1. Update and install dependencies
        sudo apt-get update
        sudo apt-get install libncurses5-dev

### 2. build
        make

### 3. launch
        ./my_sokoban 'name of the map'.txt
