## Projet Python dans le cadre du cours de Proj602_CMI : Restauration d'image.
## Bibliothèques utilisées : numpy et scipy
### Dans ce projet, nous devions effectuer une diffusion anisotropique sur des images. Dans un premier temps, nous avons utilisé la méthode de diffusion de Perona et Malik. Pour ce faire, on diffuse l'image en calculant le gradient en chaque point de l'image.Tout d'abord, il faut calculer les gradients en x et en y. Pour calculer le gradient en x, on fait la soustraction de la valeur en (x+1,y)/2 moins la valeur en (x-1,y)/2. Pour y, c'est la même chose mais avec (x,y+1)/2 et (x,y-1)/2. Pour calculer la diffusion, on applique cette formule :
### g(|∇x|) * ∇x(x+1/2,y) - g(|∇x|) * ∇x(x-1/2,y) + g(|∇y|) * ∇y(x,y+1/2) - g(|∇y|) * ∇y(x,y-1/2)
### Nous avons d'abord commencé la diffusion sur des images en niveaux de gris (fichier peronaMalik.py), puis nous avons programmé une version prenant compte des images en couleurs (fichier peronaMalikCouleurs.py)
### Afin d'avoir des résultats significatifs, nous avons effectué une diffusion sur 90 itérations avec un dt de 0.25 (résultats dans le dossier "resultats" => fichiers PeronaMalik.jpg et PeronaMalikCouleur1.jpg ).
### Nous avons également effectué un test en doublant le nombre d'itérations (180), ce qui nous donne l'image PeronaMalik2
### Ensuite, nous nous sommes intéressé à l'utilisation d'un tenseur de diffusion de Weickert. Pour cela, on applique une méthode assez similaire à celle de Perona Malik : on va calculer le gradient de chaque point de l'image qu'on va multiplier par une Gaussienne. Cela va nous donner "deux" images : une selon l'axe X et une selon l'axe Y. A partir de ces deux images, on crée un tenseur de diffusion, qui est une matrice carrée 2x2 composée de multiplications des deux images obtenues précédemment. Si on note Ix l'image selon X, Iy l'image selon Y et a, b, c et d les éléments de la matrice, on a :
### a = Ix * Ix
### b = Ix * Iy
### c = Ix * Iy
### d = Iy * Iy
### A partir de cette matrice, on peut calculer ses valeurs propres et ainsi déterminer comment diffuser le pixel de l'image en question. A noter qu'il est plus intéressant d'utiliser cette méthode sur des images couleur afin de mieux voir la diffusion. Vous pouvez voir un résultat de cette diffusion avec 5 itérations avec un dt de 0.2 (Weickert1.png) dans le dossier Résultats.
