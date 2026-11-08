# 数据核对（2026-11-08，由 scripts/data_review.py 生成）

## 1. 上场分钟

1732 个队-场中，1514 个严格在 1 分钟内（87.4%），1729 个达标或可由红牌、临时离场解释。其余 3 个逐一查了事件数据：

- 比赛 70220（Barcelona），偏差 -46.1 分钟：StatsBomb 阵容数据错误：事件里阿比达尔在下半场开始（45:00）被阿德里亚诺换下，但阵容中他的出场时段一直到终场，多算约 46 分钟（占他本季约 2%）。
- 比赛 3825873（Valencia），偏差 +11.8 分钟：真实的 10 人时段：切里舍夫 81:06 受伤离场（三个换人名额已用完），92:57 才回场；我们只少算了他回场后的约 2 分钟。
- 比赛 69222（Recreativo Huelva），偏差 -1.0 分钟：阵容与事件的换人时间相差约 30 秒，造成 13 人重叠半分钟，可忽略。

## 2. 五场比赛对照外部比赛报告

32 次换人中 29 次与外部来源一致（同一对球员、分钟相差不超过 2）。

| 比赛 | 换下 | 换上 | 我们 | 外部来源 | 一致 |
|---|---|---|---|---|---|
| 3825603 | Rafael van der Vaart | Dani Ceballos | 49′ | 49′ | 是 |
| 3825603 | Jorge Molina | Ricky van Wolfswinkel | 59′ | 60′ | 是 |
| 3825603 | Juan Domínguez | Celso Borges | 60′ | 64′ | 否 |
| 3825603 | Jonathan Rodríguez | Lucas Pérez | 64′ | 67′ | 否 |
| 3825603 | Fede Cartabia | Juanfran | 67′ | 72′ | 否 |
| 3825603 | Xavi Torres | Vincenzo Rennella | 75′ | 75′ | 是 |
| 69279 | Rafael Márquez | Martín Cáceres | 69′ | 68′ | 是 |
| 69279 | Eiður Guðjohnsen | Seydou Keita | 72′ | 72′ | 是 |
| 69279 | Samuel Eto'o | Aleksandr Hleb | 76′ | 75′ | 是 |
| 69279 | Pedro León | Fabián Canobbio | 46′ | 45′ | 是 |
| 69279 | Haris Medunjanin | Marcos Aguirre | 46′ | 45′ | 是 |
| 69279 | Jonathan Sesma | Damián Escudero | 84′ | 83′ | 是 |
| 266961 | Sandro Ramírez | Munir El Haddadi | 68′ | 68′ | 是 |
| 266961 | Ivan Rakitić | Sergi Roberto | 72′ | 72′ | 是 |
| 266961 | Jordi Alba | Jérémy Mathieu | 77′ | 76′ | 是 |
| 266961 | Luisinho | Juan Domínguez | 47′ | 46′ | 是 |
| 266961 | Jonathan Rodríguez | Luis Alberto | 58′ | 58′ | 是 |
| 266961 | Juanfran | Miguel Cardoso | 70′ | 70′ | 是 |
| 3825641 | Iñaki Williams | Eneko Bóveda | 73′ | 73′ | 是 |
| 3825641 | Markel Susaeta | Ibai Gómez | 81′ | 81′ | 是 |
| 3825641 | Beñat Etxebarria | Javi Eraso | 84′ | 84′ | 是 |
| 3825641 | Omar Mascarell | Carlos Carmona | 46′ | 45′ | 是 |
| 3825641 | Miguel Ángel Guerrero | Sergio Mendigutxia | 61′ | 61′ | 是 |
| 3825641 | Alen Halilović | Pablo Pérez | 70′ | 70′ | 是 |
| 3773587 | Ousmane Dembélé | Ansu Fati | 62′ | 62′ | 是 |
| 3773587 | Pedri | Philippe Coutinho | 62′ | 62′ | 是 |
| 3773587 | Antoine Griezmann | Francisco Trincão | 80′ | 80′ | 是 |
| 3773587 | Frenkie de Jong | Martin Braithwaite | 80′ | 80′ | 是 |
| 3773587 | Sergi Roberto | Riqui Puig | 87′ | 87′ | 是 |
| 3773587 | Jaime Mata | Enes Ünal | 71′ | 71′ | 是 |
| 3773587 | Cucho Hernández | David Timor | 89′ | 89′ | 是 |
| 3773587 | Nemanja Maksimović | Víctor Mollejo | 95′ | 95′ | 是 |

- 说明（3825603）：The three Deportivo substitutions differ by 3-5 minutes from the DFB page, whose layout is ambiguous (a stray 60' next to another substitution). StatsBomb's own substitution events (59:25, 63:39, 66:43) agree with our minutes. No third source was found. DFB may list a second Betis card for Piccini at 45'; StatsBomb has one yellow card for him (42') and he finished the match.
- 说明（3825641）：ESPN names the 61' substitute "Mendi"; StatsBomb's name is Sergio Mendigutxia Iglesias.

- 来源（3825603）：https://datencenter.dfb.de/en/data-center/la-liga/2015-2016/5-matchday/real-betis-balompie-deportivo-de-la-coruna-2226760
- 来源（69279）：https://africa.espn.com/football/match/_/gameId/253011/real-valladolid-barcelona
- 来源（266961）：https://datencenter.dfb.de/en/data-center/la-liga/2015-2016/15-matchday/fc-barcelona-deportivo-de-la-coruna-2226905
- 来源（3825641）：https://africa.espn.com/football/match/_/gameId/433894/sporting-gijon-athletic-club
- 来源（3773587）：https://africa.espn.com/football/match/_/gameId/582144/barcelona-getafe

## 3. 主要位置不明确的球员-赛季

857 个球员-赛季中有 89 个主要位置组的分钟占比低于 70%，其中 16 个出现在已报出的核心（头号或核心集合）里。按原始位置看，他们在哪两类之间切换：

| 切换 | 人数 |
|---|---|
| 边锋 / 边前卫 | 41 |
| 中前卫 / 边锋 | 14 |
| 前腰 / 边锋 | 12 |
| 边后卫 / 边锋 | 8 |
| 中前卫 / 边后卫 | 4 |
| 中锋 / 边前卫 | 4 |
| 中卫 / 后腰 | 2 |
| 前腰 / 边前卫 | 1 |
| 前腰 / 边后卫 | 1 |
| 前腰 / 中锋 | 1 |
| 后腰 / 边锋 | 1 |

出现在核心里的：

| 球队 | 赛季 | 球员 | 归入 | 占比 | 切换 | 出现在 |
|---|---|---|---|---|---|---|
| Las Palmas | 2015/16 | Jonathan Viera | MID | 53% | 边锋 33%, 中前卫 24% | att（头号）, prog |
| Celta Vigo | 2015/16 | Fabián Orellana | FWD | 55% | 边锋 52%, 前腰 33% | prog（头号） |
| Sporting Gijón | 2015/16 | Jony | FWD | 63% | 边锋 62%, 边前卫 37% | prog |
| Espanyol | 2015/16 | Marco Asensio Willemsen | MID | 62% | 前腰 39%, 边锋 35% | att（头号） |
| Levante UD | 2015/16 | José Luis Morales | MID | 58% | 边前卫 55%, 边锋 26% | att（头号） |
| Getafe | 2015/16 | Pablo Sarabia | MID | 55% | 边锋 40%, 边前卫 29% | att（头号） |
| Sporting Gijón | 2015/16 | Alen Halilović | MID | 65% | 前腰 42%, 边前卫 23% | att（头号）, prog（头号） |
| Eibar | 2015/16 | Keko | MID | 51% | 边前卫 51%, 边锋 49% | att |
| RC Deportivo La Coruña | 2015/16 | Luis Alberto | MID | 62% | 前腰 40%, 边锋 22% | att |
| Real Betis | 2015/16 | Joaquín | MID | 59% | 边前卫 42%, 边锋 36% | att（头号） |
| Real Betis | 2015/16 | Álvaro Cejudo | FWD | 53% | 边锋 53%, 边前卫 33% | prog |
| Barcelona | 2012/13 | Andrés Iniesta | MID | 66% | 中前卫 63%, 边锋 34% | att |
| Barcelona | 2008/09 | Andrés Iniesta | FWD | 53% | 边锋 53%, 中前卫 45% | link |
| Sevilla | 2015/16 | Michael Krohn-Dehli | FWD | 64% | 边锋 63%, 前腰 15% | link（头号） |
| Espanyol | 2015/16 | Burgui | FWD | 64% | 边锋 63%, 边前卫 30% | link（头号） |
| Getafe | 2015/16 | Wanderson Maciel Sousa Campos | FWD | 60% | 边锋 48%, 边前卫 19% | prog |
