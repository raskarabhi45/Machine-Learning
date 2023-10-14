#include<iostream>
using namespace std;

class base{
    public:
    base(){
        cout<<"Inside base constructor"<<endl;
    }

    void Hello(){
        cout<<"Inside hello of base"<<endl;
    }

    ~base(){
        cout<<"Inside base destructor"<<endl;
    }

};

class derived: public base
{
    public:
    derived(){
        cout<<"Inside derived constructor"<<endl;
    }

    void Hello(){
        cout<<"Inside hello of derived"<<endl;
    }
    ~derived(){
        cout<<"Inside derived destructor"<<endl;
    }

};

int main()
{
derived obj;
obj.Hello();

    return 0;
}