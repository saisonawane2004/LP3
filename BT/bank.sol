// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract BankAccount {
    // Mapping to store balances of each customer (address)
    mapping(address => uint256) private balances;

    // Event logs for transparency
    event Deposit(address indexed customer, uint256 amount);
    event Withdraw(address indexed customer, uint256 amount);

    // Deposit function (Payable to accept Ether)
    function deposit() external payable {
        require(msg.value > 0, "Deposit amount must be greater than zero");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    // Withdraw function
    function withdraw(uint256 amount) external {
        require(amount > 0, "Withdraw amount must be greater than zero");
        require(balances[msg.sender] >= amount, "Insufficient balance");

        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);

        emit Withdraw(msg.sender, amount);
    }

    // Check balance function
    function getBala() external view returns (uint256) {
        return balances[msg.sender];
    }
}
