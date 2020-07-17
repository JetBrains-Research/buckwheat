class Student {
    fullName: string;
    constructor(public firstName: string, public middleInitial: string, public lastName: string) {
        this.fullName = firstName + " " + middleInitial + " " + lastName;
    }
}

interface Person {
    firstName: string;
    lastName: string;
}

// Greeting function
function greeter(person: Person) {
    return "Hello, " + person.firstName + " " + person.lastName;
}

/** Initialize user instance of Student class */
let user = new Student("Jane", "M.", "User");

document.body.textContent = greeter(user);
