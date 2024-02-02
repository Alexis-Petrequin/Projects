/*
** EPITECH PROJECT, 2019
** my_putstr
** File description:
** my_putstr.c
*/

int stickline_error(char *line, int z);
int sticknbr_error(char *nbr, int maxsticks);
int starterror(int ac, char **av);
int letter_error(char **av, int i);
int player_line(int z);
int player_nbr(int maxsticks);
int stickgestion(char **map, int z, int maxsticks);
int fonction(char **map, int i);
int check(char **map, int z);
char **lines(int z, char **map, int nb, int sticks);
char **lineone(int z, char **map, int nb);
char **drawmap(int z, char **map);
char **create_map(char **av);
int number(char *str);
char my_putchar(char c);
int my_put_nbr(int nb);
int my_putstr(char *str);
int checkupdate(char **map, int line, int nbr, int z);
int update(char **map, int nbsticks, int linesticks, int z);
int iastickgestion(char **map, int lines, int maxsticks);
