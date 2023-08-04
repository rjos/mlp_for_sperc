# coding=utf-8

# Author: Rodolfo J. O. Soares <rodolfoj.soares@gmail.com>

import numpy as np
import re

class KeelAttribute:
    """
    A class that represent an attribute of keel dataset format.
    """
    TYPE_REAL, TYPE_INTEGER, TYPE_NOMINAL = ("real", "integer", "nominal")

    def __init__(self, attribute_name, attribute_type, attribute_range, attribute_builder):
        self.name = attribute_name
        self.type = attribute_type
        self.range = attribute_range
        self.builder = attribute_builder

class KeelDataSet:
    """
    A class that represent the keel dataset format.
    """
    UNKNOWN = '?'

    def __init__(self, relation_name, attributes, data, inputs=None, outputs=None):
        self.name = relation_name
        self.attributes = attributes
        self.data = data
        self.inputs = inputs
        self.outputs = outputs
        self.shape = len(data[0]), len(data)
        self.ir = self.__imbalance_ratio()

    def __get_data(self, attributes):
        return [self.data[self.attributes.index(a)] for a in attributes]

    def __imbalance_ratio(self):
        """Compute the imbalance ratio of the dataset
        """
        labels = self.__get_data(self.outputs)
        labels = np.concatenate(labels)

        _, count_classes = np.unique(labels, return_counts=True)
        
        max_count = np.max(count_classes)
        min_count = np.min(count_classes)

        return round((max_count / min_count), 2)

    def get_data(self):
        """Returns (data, target) of the dataset.
        """
        inputs = self.__get_data(self.inputs)
        outputs = self.__get_data(self.outputs)

        return np.transpose(inputs), np.concatenate(outputs)

    def __str__(self):
        row_format = "{:<31}" * 5

        labels = self.__get_data(self.outputs)
        labels = np.concatenate(labels)

        classes = np.unique(labels)

        # metadata = f"{self.name}:\tAttributes: {self.shape[1]}\tSamples: {self.shape[0]}\tClasses: {classes.shape[0]}\tImbalance Ratio: {self.ir}"
        return row_format.format(f"{self.name} ", *[f"Attributes: {self.shape[1]}", f"Samples: {self.shape[0]}", f"Classes: {classes.shape[0]}", f"IR: {self.ir}"])

    def __get_header(self):
        """Get the header of a keel dataset format.
        """
        header = f"@relation {self.name}\n"
        
        attributes = []
        for attr in self.attributes:
            attr_type = "real" if attr.type == KeelAttribute.TYPE_REAL else "integer" if attr.type == KeelAttribute.TYPE_INTEGER else ''
            if len(attr_type) > 0:
                attributes.append(f"@attribute {attr.name} {attr_type} [{attr.range[0]}, {attr.range[1]}]")
            else:
                attributes.append("@attribute " + attr.name + " {" + (", ").join(list(attr.range)) + "}")
        
        header += "\n".join(attributes)
        header += "\n"
        header += f"@inputs {(', ').join([attr.name for attr in self.inputs])}\n"
        header += f"@outputs {(', ').join([attr.name for attr in self.outputs])}\n"
        header += "@data\n"
        return header

    def save(self, path):
        """Export the data on keel dataset format.

        Parameters
        ----------
        path : str
               The filepath to save the dataset.
        """
        with open(path, 'w') as f:
            # Write header of database
            f.write(self.__get_header())

            # Write data of database
            data = list(map(list, zip(*self.data)))
            data = '\n'.join(map(', '.join, map(lambda x: map(str, x), data)))
            f.write(data)

def load_keel_file(path):
    """Load a keel dataset format.

    Parameters
    ----------
    path : str
           The filepath of the keel dataset format.

    Returns
    -------
    keel_dataset: KeelDataset
                  The keel dataset format loaded.
    """

    handle = open(path)
    try:
        line = handle.readline().strip()

        header_parts = line.split()
        if header_parts[0] != "@relation" or len(header_parts) != 2:
            raise SyntaxError("This is not a valid keel database.")
        
        # Get database name
        relation_name = header_parts[1]

        # Get attributes
        line = handle.readline().strip()

        attrs = []
        lkp = {}
        while line.startswith("@attribute"):
            # Get attribute name
            attr_name = line.split(" ")[1]

            # Get attribute type
            match = re.findall(r"\s([a-z]+)\s{0,1}\[", line)
            if len(match) > 0:
                attr_type = match[0]
            else:
                attr_type = "nominal"

            # Get values range
            if attr_type != "nominal":
                match = re.findall(r"\[(.*?)\]", line)
                attr_builder = float if attr_type == "real" else int
                attr_range = tuple(map(attr_builder, match[0].split(",")))
            else:
                match = re.findall(r"\{(.*?)\}", line)
                attr_builder = str
                attr_range = tuple(match[0].replace(" ", "").split(","))

            keel_attribute = KeelAttribute(attr_name, attr_type, attr_range, attr_builder)
            
            attrs.append(keel_attribute)
            lkp[attr_name] = keel_attribute

            line = handle.readline().strip()

        # Get inputs
        if not line.startswith("@input"):
            raise SyntaxError("Expected @input or @inputs. " + line)
        
        inputs_parts = line.split(maxsplit=1)
        inputs_name = inputs_parts[1].replace(" ", "").split(",")
        inputs = [lkp[name] for name in inputs_name]

        # Get output
        line = handle.readline().strip()
        if not line.startswith("@output"):
            raise SyntaxError("Expected @outputs or @outputs. " + line)
        
        output_parts = line.split(maxsplit=1)
        output_name = output_parts[1].replace(" ", "").split(",")
        outputs = [lkp[name] for name in output_name]

        # Get data
        line = handle.readline().strip()
        if line != "@data":
            raise SyntaxError("Expected @data.")
        
        data = [[] for _ in range(len(attrs))]
        for data_line in handle:
            if data_line:
                data_values = data_line.strip().replace(" ", "").split(',')
                for lst, value, attr in zip(data, data_values, attrs):
                    v = value
                    v = v if v == KeelDataSet.UNKNOWN else attr.builder(v)
                    lst.append(v)
        
        return KeelDataSet(relation_name, attrs, data, inputs, outputs)

    finally:
        if path:
            handle.close()

