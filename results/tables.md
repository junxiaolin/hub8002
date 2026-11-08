## E4 Barcelona: hubs per season (our model / plan Poisson model / MDL-CM)

| period | matches | organising: ours | Poisson | MDL-CM | threat: ours | Poisson | MDL-CM |
|---|---|---|---|---|---|---|---|
| Rijkaard 2004-08 | 77 | — | Lionel Messi, Xavi, Ronaldinho | — | — | Lionel Messi, Deco, Ronaldinho | — |
| 2008/09 | 31 | — | Andrés Iniesta, Lionel Messi, Xavi | — | Xavi | Dani Alves, Xavi | — |
| 2009/10 | 35 | — | Lionel Messi, Dmytro Chygrynskiy, Xavi | — | — | Dani Alves, Andrés Iniesta, Xavi | — |
| 2010/11 | 33 | — | Pedro, Ibrahim Afellay, Lionel Messi | — | Lionel Messi | Dani Alves, Andrés Iniesta, Lionel Messi | — |
| 2011/12 | 37 | — | Lionel Messi, Jonathan dos Santos, Xavi | — | Lionel Messi | Dani Alves, Lionel Messi, Martín Montoya | — |
| 2012/13 | 32 | — | Thiago Alcântara, Lionel Messi, Xavi | — | — | Dani Alves, Andrés Iniesta, Lionel Messi | — |
| 2013/14 | 31 | — | Dani Alves, Jordi Alba, Xavi | — | — | Dani Alves, Andrés Iniesta, Lionel Messi | — |
| 2014/15 | 38 | — | Pedro, Neymar, Lionel Messi | — | Lionel Messi | Dani Alves, Lionel Messi, Xavi | — |
| 2015/16 | 38 | — | Neymar, Lionel Messi | — | Lionel Messi | Dani Alves, Andrés Iniesta, Lionel Messi | — |
| 2016/17 | 34 | — | Neymar, Lionel Messi, Arda Turan | — | — | Andrés Iniesta, Lionel Messi, Sergi Roberto | — |
| 2017/18 | 36 | — | Sergio Busquets, Ivan Rakitić, Lionel Messi | — | — | Jordi Alba, Andrés Iniesta, Lionel Messi | — |
| 2018/19 | 34 | — | Philippe Coutinho, Ousmane Dembélé, Lionel Messi | — | — | Jordi Alba, Lionel Messi, Sergi Roberto | — |
| 2019/20 | 33 | — | Sergio Busquets, Ousmane Dembélé, Lionel Messi | — | — | Jordi Alba, Lionel Messi, Sergi Roberto | — |
| 2020/21 | 35 | — | Sergio Busquets, Lionel Messi, Miralem Pjanić | — | Lionel Messi | Jordi Alba, Lionel Messi, Sergi Roberto | — |

## E1 false-positive rate (organising layer, phi measured)

| method | het | equal | real | rest |
|---|---|---|---|---|
| Our model | none | 0.00 | 0.01 | 0.01 |
| Plan Poisson model | none | 0.02 | 0.02 | 0.05 |
| No position | none | 0.00 | 0.00 | 0.00 |
| No minutes | none | 0.00 | 1.00 | 1.00 |
| No shared rho | none | 0.00 | 0.01 | 0.02 |
| MDL-CM | none | 1.00 | 0.00 | 0.00 |
| MDL-ER | none | 0.00 | 1.00 | 1.00 |
| MDL-CM (per-minute) | none | 1.00 | 1.00 | 0.96 |
| MDL-ER (per-minute) | none | 0.00 | 0.07 | 0.11 |
| Most passes | none | 1.00 | 1.00 | 1.00 |
| Most per 90 | none | 1.00 | 1.00 | 1.00 |
| Top PageRank | none | 1.00 | 1.00 | 1.00 |
| Mean + 2SD | none | 0.04 | 0.41 | 0.57 |
| Average | none | 1.00 | 1.00 | 1.00 |
| Loubar | none | 1.00 | 1.00 | 1.00 |
| Our model | measured | 0.04 | 0.06 | 0.05 |
| Plan Poisson model | measured | 0.99 | 0.98 | 0.96 |
| No position | measured | 0.04 | 0.07 | 0.06 |
| No minutes | measured | 0.04 | 1.00 | 1.00 |
| No shared rho | measured | 0.19 | 0.24 | 0.23 |
| MDL-CM | measured | 0.89 | 0.00 | 0.00 |
| MDL-ER | measured | 0.34 | 1.00 | 1.00 |
| MDL-CM (per-minute) | measured | 0.89 | 0.80 | 0.74 |
| MDL-ER (per-minute) | measured | 0.34 | 0.49 | 0.52 |
| Most passes | measured | 1.00 | 1.00 | 1.00 |
| Most per 90 | measured | 1.00 | 1.00 | 1.00 |
| Top PageRank | measured | 1.00 | 1.00 | 1.00 |
| Mean + 2SD | measured | 0.61 | 0.65 | 0.69 |
| Average | measured | 1.00 | 1.00 | 1.00 |
| Loubar | measured | 1.00 | 1.00 | 1.00 |

## E1 detection / exact recovery (real rotation, measured phi and individual differences)

| method | group | rho | detection | exact |
|---|---|---|---|---|
| Our model | MID | 1.25 | 0.04 | 0.03 |
| Our model | MID | 1.5 | 0.12 | 0.09 |
| Our model | MID | 2.0 | 0.50 | 0.46 |
| Our model | FWD | 1.25 | 0.03 | 0.02 |
| Our model | FWD | 1.5 | 0.10 | 0.08 |
| Our model | FWD | 2.0 | 0.40 | 0.35 |
| Plan Poisson model | MID | 1.25 | 0.62 | 0.04 |
| Plan Poisson model | MID | 1.5 | 0.76 | 0.08 |
| Plan Poisson model | MID | 2.0 | 0.97 | 0.27 |
| Plan Poisson model | FWD | 1.25 | 0.44 | 0.02 |
| Plan Poisson model | FWD | 1.5 | 0.60 | 0.08 |
| Plan Poisson model | FWD | 2.0 | 0.87 | 0.22 |
| No position | MID | 1.25 | 0.09 | 0.07 |
| No position | MID | 1.5 | 0.28 | 0.22 |
| No position | MID | 2.0 | 0.69 | 0.60 |
| No position | FWD | 1.25 | 0.00 | 0.00 |
| No position | FWD | 1.5 | 0.00 | 0.00 |
| No position | FWD | 2.0 | 0.04 | 0.02 |
| No minutes | MID | 1.25 | 0.18 | 0.00 |
| No minutes | MID | 1.5 | 0.26 | 0.00 |
| No minutes | MID | 2.0 | 0.48 | 0.00 |
| No minutes | FWD | 1.25 | 0.15 | 0.00 |
| No minutes | FWD | 1.5 | 0.20 | 0.00 |
| No minutes | FWD | 2.0 | 0.29 | 0.00 |
| No shared rho | MID | 1.25 | 0.10 | 0.05 |
| No shared rho | MID | 1.5 | 0.25 | 0.14 |
| No shared rho | MID | 2.0 | 0.67 | 0.49 |
| No shared rho | FWD | 1.25 | 0.13 | 0.06 |
| No shared rho | FWD | 1.5 | 0.31 | 0.17 |
| No shared rho | FWD | 2.0 | 0.63 | 0.39 |
| MDL-CM | MID | 1.25 | 0.00 | 0.00 |
| MDL-CM | MID | 1.5 | 0.00 | 0.00 |
| MDL-CM | MID | 2.0 | 0.00 | 0.00 |
| MDL-CM | FWD | 1.25 | 0.00 | 0.00 |
| MDL-CM | FWD | 1.5 | 0.00 | 0.00 |
| MDL-CM | FWD | 2.0 | 0.00 | 0.00 |
| MDL-ER | MID | 1.25 | 1.00 | 0.00 |
| MDL-ER | MID | 1.5 | 1.00 | 0.00 |
| MDL-ER | MID | 2.0 | 1.00 | 0.00 |
| MDL-ER | FWD | 1.25 | 0.85 | 0.00 |
| MDL-ER | FWD | 1.5 | 0.89 | 0.00 |
| MDL-ER | FWD | 2.0 | 0.95 | 0.00 |
| MDL-CM (per-minute) | MID | 1.25 | 0.33 | 0.21 |
| MDL-CM (per-minute) | MID | 1.5 | 0.51 | 0.30 |
| MDL-CM (per-minute) | MID | 2.0 | 0.73 | 0.48 |
| MDL-CM (per-minute) | FWD | 1.25 | 0.01 | 0.01 |
| MDL-CM (per-minute) | FWD | 1.5 | 0.03 | 0.03 |
| MDL-CM (per-minute) | FWD | 2.0 | 0.19 | 0.13 |
| MDL-ER (per-minute) | MID | 1.25 | 0.48 | 0.01 |
| MDL-ER (per-minute) | MID | 1.5 | 0.62 | 0.04 |
| MDL-ER (per-minute) | MID | 2.0 | 0.85 | 0.24 |
| MDL-ER (per-minute) | FWD | 1.25 | 0.24 | 0.00 |
| MDL-ER (per-minute) | FWD | 1.5 | 0.29 | 0.00 |
| MDL-ER (per-minute) | FWD | 2.0 | 0.41 | 0.01 |
| Most passes | MID | 1.25 | 0.40 | 0.40 |
| Most passes | MID | 1.5 | 0.54 | 0.54 |
| Most passes | MID | 2.0 | 0.79 | 0.79 |
| Most passes | FWD | 1.25 | 0.03 | 0.03 |
| Most passes | FWD | 1.5 | 0.05 | 0.05 |
| Most passes | FWD | 2.0 | 0.19 | 0.19 |
| Most per 90 | MID | 1.25 | 0.35 | 0.35 |
| Most per 90 | MID | 1.5 | 0.58 | 0.58 |
| Most per 90 | MID | 2.0 | 0.90 | 0.90 |
| Most per 90 | FWD | 1.25 | 0.01 | 0.01 |
| Most per 90 | FWD | 1.5 | 0.04 | 0.04 |
| Most per 90 | FWD | 2.0 | 0.21 | 0.21 |
| Top PageRank | MID | 1.25 | 0.10 | 0.10 |
| Top PageRank | MID | 1.5 | 0.08 | 0.08 |
| Top PageRank | MID | 2.0 | 0.06 | 0.06 |
| Top PageRank | FWD | 1.25 | 0.07 | 0.07 |
| Top PageRank | FWD | 1.5 | 0.07 | 0.07 |
| Top PageRank | FWD | 2.0 | 0.05 | 0.05 |
| Mean + 2SD | MID | 1.25 | 0.37 | 0.28 |
| Mean + 2SD | MID | 1.5 | 0.53 | 0.42 |
| Mean + 2SD | MID | 2.0 | 0.80 | 0.69 |
| Mean + 2SD | FWD | 1.25 | 0.01 | 0.01 |
| Mean + 2SD | FWD | 1.5 | 0.02 | 0.02 |
| Mean + 2SD | FWD | 2.0 | 0.15 | 0.12 |
| Average | MID | 1.25 | 0.96 | 0.00 |
| Average | MID | 1.5 | 0.99 | 0.00 |
| Average | MID | 2.0 | 1.00 | 0.00 |
| Average | FWD | 1.25 | 0.56 | 0.00 |
| Average | FWD | 1.5 | 0.69 | 0.00 |
| Average | FWD | 2.0 | 0.83 | 0.00 |
| Loubar | MID | 1.25 | 0.94 | 0.00 |
| Loubar | MID | 1.5 | 0.95 | 0.00 |
| Loubar | MID | 2.0 | 0.99 | 0.00 |
| Loubar | FWD | 1.25 | 0.49 | 0.00 |
| Loubar | FWD | 1.5 | 0.61 | 0.00 |
| Loubar | FWD | 2.0 | 0.77 | 0.00 |

## Table 1: matches needed (detection >= 80%, false positives <= 5%; "-" = not reached within 38)

| world | method | rho | org MID | org FWD | box MID | box FWD |
|---|---|---|---|---|---|---|
| individual differences | Our model | 1.25 | - | - | - | - |
| individual differences | Our model | 1.5 | - | - | - | - |
| individual differences | Our model | 2.0 | - | - | - | - |
| individual differences | Our model | 3.0 | 19 | 38 | - | - |
| individual differences | Plan Poisson model | 1.25 | - | - | - | - |
| individual differences | Plan Poisson model | 1.5 | - | - | - | - |
| individual differences | Plan Poisson model | 2.0 | - | - | - | - |
| individual differences | Plan Poisson model | 3.0 | - | - | - | - |
| individual differences | MDL-CM (per-minute) | 1.25 | - | - | - | - |
| individual differences | MDL-CM (per-minute) | 1.5 | - | - | - | - |
| individual differences | MDL-CM (per-minute) | 2.0 | - | - | - | - |
| individual differences | MDL-CM (per-minute) | 3.0 | - | - | - | - |
| identical players | Our model | 1.25 | - | - | - | - |
| identical players | Our model | 1.5 | - | - | - | - |
| identical players | Our model | 2.0 | - | - | - | - |
| identical players | Our model | 3.0 | 3 | 5 | - | - |
| identical players | Plan Poisson model | 1.25 | - | - | - | - |
| identical players | Plan Poisson model | 1.5 | 19 | 38 | - | - |
| identical players | Plan Poisson model | 2.0 | 5 | 8 | - | - |
| identical players | Plan Poisson model | 3.0 | 5 | 5 | 38 | 38 |
| identical players | MDL-CM (per-minute) | 1.25 | - | - | - | - |
| identical players | MDL-CM (per-minute) | 1.5 | - | - | - | - |
| identical players | MDL-CM (per-minute) | 2.0 | - | - | - | - |
| identical players | MDL-CM (per-minute) | 3.0 | - | - | - | - |

## Table 2a: E3 first half -> second half (gain in log predictive density over "no hub", mean per unit, 95% CI)

**org** (34 units; saturated reference gain -3.7)

| method | gain | 95% CI | Jaccard H1-H2 | units with a hub |
|---|---|---|---|---|
| Our model | 0.22 | [0.00, 0.59] | 0.25 | 0.06 |
| Plan Poisson model | 1.55 | [0.91, 2.18] | 0.43 | 1.00 |
| No position | 0.00 | [0.00, 0.00] | 0.00 | 0.00 |
| No minutes | 0.50 | [0.09, 0.96] | 0.39 | 1.00 |
| No shared rho | 0.63 | [0.18, 1.18] | 0.25 | 0.32 |
| MDL-CM | 0.00 | [0.00, 0.00] | — | 0.00 |
| MDL-ER | 0.02 | [-0.37, 0.48] | 0.61 | 1.00 |
| MDL-CM (per-minute) | 0.22 | [0.02, 0.43] | 0.23 | 0.74 |
| MDL-ER (per-minute) | 0.39 | [0.08, 0.80] | 0.18 | 0.29 |
| Most passes | 0.24 | [0.05, 0.43] | 0.56 | 1.00 |
| Most per 90 | 0.18 | [-0.05, 0.44] | 0.50 | 1.00 |
| Top PageRank | 0.32 | [0.08, 0.56] | 0.62 | 1.00 |
| Mean + 2SD | 0.19 | [0.04, 0.34] | 0.32 | 0.62 |
| Average | 0.39 | [0.06, 0.76] | 0.62 | 1.00 |
| Loubar | 0.36 | [-0.00, 0.74] | 0.59 | 1.00 |

**box** (34 units; saturated reference gain 7.5)

| method | gain | 95% CI | Jaccard H1-H2 | units with a hub |
|---|---|---|---|---|
| Our model | 0.41 | [0.12, 0.74] | 0.14 | 0.18 |
| Plan Poisson model | 2.11 | [1.52, 2.71] | 0.43 | 0.91 |
| No position | 0.00 | [0.00, 0.00] | — | 0.00 |
| No minutes | 1.29 | [0.78, 1.84] | 0.43 | 0.62 |
| No shared rho | 1.40 | [0.80, 2.00] | 0.42 | 0.56 |
| MDL-CM | 0.00 | [0.00, 0.00] | — | 0.00 |
| MDL-ER | 2.04 | [1.44, 2.66] | 0.58 | 1.00 |
| MDL-CM (per-minute) | 0.00 | [0.00, 0.00] | — | 0.00 |
| MDL-ER (per-minute) | 2.28 | [1.63, 2.95] | 0.50 | 1.00 |
| Most passes | 0.96 | [0.55, 1.37] | 0.74 | 1.00 |
| Most per 90 | 0.72 | [0.28, 1.21] | 0.50 | 1.00 |
| Top PageRank | 0.17 | [-0.17, 0.55] | 0.53 | 1.00 |
| Mean + 2SD | 1.30 | [0.86, 1.72] | 0.67 | 0.94 |
| Average | 1.93 | [1.30, 2.64] | 0.59 | 1.00 |
| Loubar | 2.04 | [1.39, 2.73] | 0.49 | 1.00 |


## E5 La Liga 2015/16: hubs per team

| team | organising: ours | Poisson | MDL-CM | MDL-CM (per-min) | threat: ours | Poisson |
|---|---|---|---|---|---|---|
| Athletic Club | — | Mikel San José (MID), Beñat Etxebarria (MID), Iker Muniain (FWD) | — | — | — | Beñat Etxebarria, De Marcos, Mikel Balenziaga |
| Atlético Madrid | Filipe Luis (DEF) | Antoine Griezmann (FWD), Filipe Luis (DEF), Luciano Vietto (FWD) | — | Koke (MID), Gabi (MID), Tiago (MID) | — | Jesús Gámez, Filipe Luis, Juanfran |
| Barcelona | — | Neymar (FWD), Lionel Messi (FWD) | — | — | Lionel Messi | Dani Alves, Andrés Iniesta, Lionel Messi |
| Celta Vigo | Andreu Fontàs (DEF), Fabián Orellana (FWD) | Andreu Fontàs (DEF), Fabián Orellana (FWD), Augusto Fernández (MID) | — | — | Nolito | Nolito, Jonny Castro, Hugo Mallo |
| Eibar | — | Dani García (MID), Iván Ramis (DEF), Lillo Castellano (DEF) | — | Dani García (MID) | — | Ander Capa, Keko, David Juncà |
| Espanyol | — | Burgui (FWD), Joan Jordán (MID), Hernán Pérez (FWD) | — | — | — | Marco Asensio Willemsen, Burgui, Salva Sevilla |
| Getafe | — | Mehdi Lacen (MID), Pedro León (FWD), Wanderson Maciel Sousa Campos (FWD) | — | Mehdi Lacen (MID) | — | Damián Suárez, Pablo Sarabia, Abdoul Karim Yoda |
| Granada | — | Abdoulaye Doucouré (MID), Rubén Pérez (MID), Fran Rico (MID) | — | Abdoulaye Doucouré (MID), Rubén Pérez (MID), Fran Rico (MID) | — | Cristiano Biraghi, Miguel Lopes, Fran Rico |
| Las Palmas | — | Roque Mesa (MID), Vicente Gómez (MID), Juan Carlos Valerón (MID) | — | Juan Carlos Valerón (MID) | — | Momo, David Simón, Jonathan Viera |
| Levante UD | — | Víctor Camarasa (MID), Verza (MID), Joan Verdú (MID) | — | — | — | José Luis Morales, Toño, Joan Verdú |
| Málaga | — | Recio (MID), Pablo Fornals (MID), Ignacio Camacho (MID) | — | Fábio Espinho (FWD) | — | Roberto Rosales, Federico Ricca, Arthur Boka |
| RC Deportivo La Coruña | — | Fede Cartabia (FWD), Pedro Mosquera (MID), Cani (FWD) | — | Pedro Mosquera (MID) | — | Fayçal Fajr, Juanfran, Luis Alberto |
| Rayo Vallecano | — | Patrick Ebert (FWD), Zé Castro (DEF), Roberto Trashorras (MID) | — | Roberto Trashorras (MID) | — | Bebé, Răzvan Raț, Adri Embarba |
| Real Betis | — | Dani Ceballos (MID), Francisco Portillo (MID), Álvaro Cejudo (FWD) | — | Francisco Portillo (MID) | — | Joaquín, Dani Ceballos, Cristiano Piccini |
| Real Madrid | — | Lucas Vázquez (FWD), Luka Modrić (MID), Toni Kroos (MID) | — | Toni Kroos (MID) | — | Marcelo, James Rodríguez, Daniel Carvajal |
| Real Sociedad | — | Carlos Vela (FWD), David Zurutuza (FWD), Bruma (FWD) | — | Asier Illarramendi (MID) | — | Yuri Berchiche, Carlos Vela, Xabi Prieto |
| Sevilla | — | Éver Banega (MID), Michael Krohn-Dehli (FWD), Kolo (DEF) | — | — | Éver Banega | Éver Banega, Sergio Escudero, Mariano |
| Sporting Gijón | — | Luis Hernández (DEF), Carlos Carmona (FWD), Alberto Lora (DEF) | — | Nacho Cases (MID) | — | Jony, Isma López, Alberto Lora |
| Valencia | — | Shkodran Mustafi (DEF), Dani Parejo (MID), Javi Fuego (MID) | — | Dani Parejo (MID) | — | José Gayá, Antonio Barragán, João Cancelo |
| Villarreal | — | Jonathan dos Santos (MID), Manu Trigueros (MID), Bruno Soriano (MID) | — | Rodri (MID) | — | Adrián López, Léo Baptistão, Mario Gaspar |

## Table 2b: E6 within-position permutation (share of permutations with >= 1 hub, mean over units)

| method | organising | threat |
|---|---|---|
| Our model | 0.012 | 0.005 |
| Plan Poisson model | 0.155 | 0.213 |
| MDL-CM (per-minute) | 0.942 | 0.140 |
| MDL-CM | 0.001 | 0.001 |

## E7 confirmation time

| layer | hubs at season end | stay > 0.9 | median share of season (0.9) | stay > 0.5 | median share of season (0.5) |
|---|---|---|---|---|---|
| org | 3 | 1 | 0.58 | 3 | 0.32 |
| box | 8 | 0 | — | 8 | 0.33 |

| unit | layer | hub | final probability | first match staying > 0.5 | > 0.9 |
|---|---|---|---|---|---|
| Barcelona 2008/09 | box | Xavi | 0.60 | 30 / 31 | None |
| Barcelona 2010/11 | box | Lionel Messi | 0.61 | 30 / 33 | None |
| Barcelona 2011/12 | box | Lionel Messi | 0.80 | 4 / 37 | None |
| Barcelona 2014/15 | box | Lionel Messi | 0.75 | 2 / 38 | None |
| Barcelona 2015/16 | box | Lionel Messi | 0.83 | 5 / 38 | None |
| Barcelona 2020/21 | box | Lionel Messi | 0.76 | 9 / 35 | None |
| Atlético Madrid 2015/16 | org | Filipe Luis | 0.97 | 12 / 38 | 22 |
| Celta Vigo 2015/16 | org | Andreu Fontàs | 0.89 | 3 / 38 | None |
| Celta Vigo 2015/16 | org | Fabián Orellana | 0.56 | 29 / 38 | None |
| Celta Vigo 2015/16 | box | Nolito | 0.63 | 15 / 38 | None |
| Sevilla 2015/16 | box | Éver Banega | 0.75 | 17 / 38 | None |

## Calibration (La Liga 2015/16)

| layer | group | rate per 90 | individual spread alpha (CV) | between-player dispersion of season totals |
|---|---|---|---|---|
| org | DEF | 30.62 | 75.4 (0.12) | 13.6 x Poisson |
| org | MID | 36.33 | 18.9 (0.23) | 33.3 x Poisson |
| org | FWD | 20.29 | 15.0 (0.26) | 34.4 x Poisson |
| org | per-match phi | 3.19 | | |
| box | DEF | 0.47 | 0.8 (1.15) | 8.7 x Poisson |
| box | MID | 0.79 | 3.8 (0.51) | 6.3 x Poisson |
| box | FWD | 1.05 | 6.0 (0.41) | 5.5 x Poisson |
| box | per-match phi | 1.11 | | |

## Sensitivity (our model on E4 + E5 units)

| variant | organising hubs | threat hubs | units unchanged | Messi threat-hub seasons | Xavi organising-hub seasons |
|---|---|---|---|---|---|
| base | 3 | 8 | 100% | 5 | 0 |
| rho_max=2 | 3 | 0 | 88% | 0 | 0 |
| rho_max=10 | 3 | 7 | 98% | 4 | 0 |
| prior shape 0.5 | 3 | 7 | 98% | 4 | 0 |
| prior shape 2 | 3 | 8 | 97% | 6 | 0 |
| prior mean x0.5 | 4 | 12 | 94% | 8 | 0 |
| prior mean x2 | 3 | 6 | 97% | 4 | 0 |
| min minutes 450 | 4 | 8 | 95% | 5 | 0 |
| s_max=4 | 3 | 8 | 100% | 5 | 0 |
| no phi correction | 12 | 8 | 91% | 5 | 0 |
| alpha x0.5 | 0 | 0 | 85% | 0 | 0 |
| alpha x2 | 23 | 33 | 65% | 10 | 1 |
| wingers in midfield | 7 | 8 | 82% | 5 | 0 |
| six roles | 14 | 32 | 68% | 6 | 0 |
