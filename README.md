# CameraTrackingHandball
Le but de ce github est de tester des méthodes pour repérer un terrain de handball dans l'espace à partir d'une vidéo et de tracker les joueurs dans le but de mesurer la distacne qu'ils parcourent.


***VisualOdometryPython***

Cette partie contient un code python faisant de l'odométrie visuelle et une triangulation pour reconstruire un nuage de point 3D représentant le terrain. Il a besoin d'une librairie associé et des poids d'un réseau yolov8, ainsi qu'une vidéo et d'un fichier contenant la calibration caméra; Il suffit d'exécuter le code pour le faire tourner. Pour ne pas tenit compte des nuages de point3D, il suffit de commenter la ligne  correspondante dans le main. Ce code est une adaptation d'un [git](https://github.com/niconielsen32/ComputerVision/tree/master/VisualOdometry)

***ROS ***

Ce git contient des noeuds ROS, il suffit de les télécharger et de recompiler le catkin_ws, pas besoin de compiler les noeuds car ils sont en python. Il faudrait passer du temps pour les convertir en c++ pour les optimiser en temps.

- Pour lancer la lecture d'une vidéo, il faut soit utilsier le noeud ROS usb_cam (qu'il faut télécharger) et adapter le device d'entrée dans le fichier yaml et le topic de sortie. Soit utiliser le noeud video_publisher et choisir le topic sur lequel publier des images. 

- my_usb_cam_viewer est un noeud qui permet de voir l'image publier sur un certain topic, il faut adapter le code pour indiquer le topic que vous souhiater visionner (ce noeud peut être remplacer par l'utilisation de Rviz)

- visual_odometry est un noeud implémentant un système d'odométrie visuelle et qui renvoie une pose de caméra sur le topic /camera/pose. Ce code à besoin d'un fichier pour la calibration de la caméra. Cette calibration peut être faite avec le noeud usb_cam et un quadrillage tel qu'un échiquier. 

- yolo_node est un noeud prenant en entrée une image et qui utlise un réseau yolov8 pour renvoyer les bounding box des joueurs détectés dans l'image




    Pour exécuter les noeuds il faut utiliser : rosrun \<nom du package> \<nom du code>. Sauf si il y a un fichier launch inclut, dans ce cas il faut faire : roslaunch \<nom du package> \<nom du launch>


***CameraMatrixMotor***

Cette partie impléméente une classe permettant de simplifier la gestion de la matrice caméra. Elle peut notamment servir à chercher la matrice caméra initiale à l'aide du code joint. On peut chercher à plaquer un terrain standard sur l'image de la vidéo avec le clavier les touches "z,s,q,d,l,m" permettent de gérer les rotations. 






"""  
Note pour plus tard  
Take exemple of : Sports Camera Calibration via Synthetic Data
https://arxiv.org/pdf/1810.10658.pdf  
"""

