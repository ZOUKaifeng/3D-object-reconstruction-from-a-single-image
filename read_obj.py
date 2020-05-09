import os
import numpy as np
class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        ##
        try:
            f = open(fileName)
           
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex_ =np.array ([round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2)])
                    self.vertices.append(vertex_)
              
                
                elif line[0] == "f":

                    string = line.replace("//", "/")
                    ##
                    
                    line_ = line.split(' ')
         
                    face = np.zeros(3)
                    
                    for i in range(3):
                        temp = line_[i+1].split('/')
                        face[i] = int(temp[0])
                    self.faces.append(face)

                    
            f.close()
            self.faces = np.asarray(self.faces).astype('int')
            self.vertices = np.asarray(self.vertices).astype('float')
        except IOError:
            print(".obj file not found.")