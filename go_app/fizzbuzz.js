/**
 * FizzBuzz implementation
 * Prints numbers from 1 to 100
 * For multiples of 3, print "Fizz" instead of the number
 * For multiples of 5, print "Buzz" instead of the number
 * For multiples of both 3 and 5, print "FizzBuzz" instead of the number
 */

function fizzBuzz(limit = 100) {
  for (let i = 1; i <= limit; i++) {
    let output = '';
    
    if (i % 3 === 0) output += 'Fizz';
    if (i % 5 === 0) output += 'Buzz';
    
    console.log(output || i);
  }
}

// Execute the FizzBuzz function
console.log('Running FizzBuzz:');
fizzBuzz();