# Simulation-Nodes

Para cada nodo se debe crear el paquete de ros con el comando:

  catkin_create_pkg pkg_name std_msgs rospy

A continuación de std_msgs se deben incluir todas las librerías de mensajes que tenga el archivo de python del paquete.

Después se compila con:

  catkin build
