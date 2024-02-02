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

int verifline(char **map, int line)
{
    int verif = 0;
    for (int i = 0; map[line][i]; ++i)
        if (map[line][i] == '|')
            ++verif;
    return (verif);
}

int iastickgestion(char **map, int lines, int maxsticks)
{
    int line = rand() % lines + 1;
    int stick = rand() % maxsticks + 1;
    int verif = 0;

    verif = verifline(map, line);
    if (verif == 0)
        while (verif == 0) {
            line = rand() % lines + 1;
            verif = verifline(map, line);
        }
    if (stick > maxsticks || stick > verif)
        while (stick > maxsticks || stick > verif) {
            stick = rand() % maxsticks + 1;
        }
    iaupdate(map, line, stick, lines);
}
