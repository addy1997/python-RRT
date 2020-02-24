import random

class Writer():

    def __is_edge(self,word1, word2):  # examine those two words whether they have edge or not
        count = 0
        for letter in word1:
            if letter in word2:
                count += 1
                word2 = word2.replace(letter, '', 1)  # remove the counting element to not counts again

        if count == len(word1) - 3:  # 3 letters are different
            return True
        else:
            return False

    def ident_vert(self,amount):
        file2 = open("words7.txt", "r")
        words = file2.read().strip().split()
        file2.close()
        vertices = random.sample(words, amount)  # the stament that determines # of vertices in the graph
        return vertices

    def write_to(self,filename,amount):
        vertices=self.ident_vert(amount)
        file = open(filename, "w")
        file.writelines("graph = {\n")
        for i in vertices:
            file.writelines("\"" + i + "\":[")
            for y in vertices:
                if i == y:
                    continue
                else:
                    if self.__is_edge(i, y):
                        file.writelines("\"" + y + "\",")
            file.writelines("],\n")
        file.writelines("}")
        file.close()
