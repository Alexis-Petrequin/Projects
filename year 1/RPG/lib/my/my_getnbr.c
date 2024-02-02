/*
** EPITECH PROJECT, 2019
** my_getnbr
** File description:
** get number
*/

int my_getnbr(char const *str)
{
    int y = 0;
    int i = 0;
    long nb = 0;
    int nbr = 0;

    for (; str[i] == '-' || str[i] == '+'; i++, y++);
    for (; str[i] >= '0' && str[i] <= '9'; i++) {
        nb += str[i] - 48;
        if (str[i + 1] >= '0' && str[i + 1] <= '9')
            nb = nb * 10;
        if (nb > 2147483648)
            return (0);
    }
    if (str[y - 1] == '-' && nb <= 2147483648)
        nbr = -nb;
    else if (nb <= 2147483647)
        nbr = nb;
    return (nbr);
}
