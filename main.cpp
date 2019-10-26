#include <iostream>
#include <cstdlib> //srand, rand
#include <ctime> //time
#include <vector> //vector

template <typename arType> class Array
{
private:
	arType* data;
	int length;
public:
	Array() {};
	Array(int _length) : length(_length)
	{
		data = new arType[length];
	}
	~Array()
	{
		delete[] data;
	}

	arType* getData()
	{
		return data;
	}
	
	int getLength()
	{
		return length;
	}

	void generateData(arType min, arType max)
	{
		srand(time(0));
		for (int i = 0; i < length; i++)
		{
			data[i] = (arType)( min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min))));
		}
	}

	void printData()
	{
		for (int i = 0; i < length; i++)
		{
			std::cout << "data[" << i << "]: " << data[i] << std::endl;
		}
	}

	void swap(int ind1, int ind2)
	{
		arType buff = data[ind1];
		data[ind1] = data[ind2];
		data[ind2] = buff;
	}
};

int clamp(int value, int min, int max)
{
	if (value > max)
	{
		return max;
	}
	else if (value < min)
	{
		return min;
	}
	return value;
}

template <typename arType>
void BubbleSort(Array<arType> &A)
{
	for (int i = 0; i < A.getLength() - 1; i++)
	{
		for (int j = 0; j < (A.getLength() - 1) - i; j++)
		{
			if (A.getData()[j] > A.getData()[j + 1])
			{
				A.swap(j, j + 1);
			}
		}
	}
}


int main(int* argc, char** argv)
{
	Array<int> A(10);
	A.generateData(0, 20);
	A.printData();
	std::cout << "\nSorted:\n";
	BubbleSort(A);
	A.printData();
	system("pause");
}