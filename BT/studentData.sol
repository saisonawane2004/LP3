// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract StudentData {

    // Structure to store Student details
    struct Student {
        uint256 rollNo;
        string name;
        uint8 age;
        string course;
    }

    // Array to store multiple students
    Student[] public students;

    // Event to log addition of student
    event StudentAdded(uint256 rollNo, string name);

    // Function to add a new student
    function addStudent(uint256 _rollNo, string memory _name, uint8 _age, string memory _course) public {
        Student memory newStudent = Student({
            rollNo: _rollNo,
            name: _name,
            age: _age,
            course: _course
        });

        students.push(newStudent);
        emit StudentAdded(_rollNo, _name);
    }

    // Function to get total number of students
    function getTotalStudents() public view returns (uint256) {
        return students.length;
    }

    // Function to get student details by index
    function getStudent(uint256 index) public view returns (uint256, string memory, uint8, string memory) {
        require(index < students.length, "Invalid student index");
        Student memory s = students[index];
        return (s.rollNo, s.name, s.age, s.course);
    }

    // Fallback function (called when no other function matches)
    fallback() external payable {
        // If someone sends Ether or wrong function call
        // Just receive the Ether and do nothing
    }

    // Receive Ether (optional for accepting ETH transfers)
    receive() external payable {}
}
