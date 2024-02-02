/*
** EPITECH PROJECT, 2019
** my_putstr
** File description:
** my_putstr.c
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "include.h"

int matchstick(char **map, char **av)
{
    int stillsticks = 1;
    int test = 0;
    stillsticks = check(map, number(av[1]));
    while (stillsticks != 0) {
        test = stickgestion(map, number(av[1]), number(av[2]));
        stillsticks = check(map, number(av[1]));
        if (stillsticks == 0) {
            my_putstr("You lost, too bad...\n");
            return (2);
        } else
            my_putstr("\n");
        if (test == 0)
            return (0);
        if (test == 84)
            return (84);
        else if (test == 101);
        else
            iastickgestion(map, number(av[1]), number(av[2]));
        stillsticks = check(map, number(av[1]));
        if (stillsticks == 0){
            my_putstr("I lost... snif... but I'll get you next time !!\n");
            return (1);
        } else
            my_putstr("\n");
    }
}

int main(int ac, char **av)
{
    char **map;
    int error = starterror (ac, av);
    int result = 0;

    if (error == 84)
        return (84);
    map = create_map(av);
    result = matchstick(map, av);
    return (result);
}