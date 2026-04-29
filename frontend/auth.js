// Authentication System for SELA
// Uses localStorage for demo purposes (replace with backend API in production)

// Check if user is already logged in
function checkAuth() {
    const currentUser = localStorage.getItem('selaCurrentUser');
    const currentPage = window.location.pathname;
    
    if (currentUser && (currentPage.includes('login.html') || currentPage.includes('register.html'))) {
        // User is logged in, redirect to main app
        window.location.href = 'index.html';
    } else if (!currentUser && currentPage.includes('index.html')) {
        // User is not logged in, redirect to login
        window.location.href = 'login.html';
    }
}

// Run auth check on page load
checkAuth();

// Login Form Handler
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const remember = document.getElementById('remember').checked;
        
        // Get users from localStorage
        const users = JSON.parse(localStorage.getItem('selaUsers') || '[]');
        
        // Find user (case-insensitive username)
        const user = users.find(u => u.username.toLowerCase() === username.toLowerCase() && u.password === password);
        
        if (user) {
            // Login successful
            const userData = {
                username: user.username,
                fullname: user.fullname,
                email: user.email,
                loginTime: new Date().toISOString()
            };
            
            localStorage.setItem('selaCurrentUser', JSON.stringify(userData));
            
            if (remember) {
                localStorage.setItem('selaRememberMe', 'true');
            }
            
            // Redirect to main app
            window.location.href = 'index.html';
        } else {
            // Login failed
            showError('Invalid username or password');
        }
    });
}

// Register Form Handler
const registerForm = document.getElementById('registerForm');
if (registerForm) {
    registerForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fullname = document.getElementById('fullname').value;
        const email = document.getElementById('email').value;
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const terms = document.getElementById('terms').checked;
        
        // Validation
        if (password !== confirmPassword) {
            showError('Passwords do not match');
            return;
        }
        
        if (password.length < 6) {
            showError('Password must be at least 6 characters');
            return;
        }
        
        if (!terms) {
            showError('You must agree to the Terms of Service');
            return;
        }
        
        // Get existing users
        const users = JSON.parse(localStorage.getItem('selaUsers') || '[]');
        
        // Check if username already exists
        if (users.find(u => u.username === username)) {
            showError('Username already exists');
            return;
        }
        
        // Check if email already exists
        if (users.find(u => u.email === email)) {
            showError('Email already registered');
            return;
        }
        
        // Create new user
        const newUser = {
            fullname,
            email,
            username,
            password, // In production, hash this!
            createdAt: new Date().toISOString()
        };
        
        users.push(newUser);
        localStorage.setItem('selaUsers', JSON.stringify(users));
        
        // Show success message
        showSuccess('Account created successfully! Redirecting to login...');
        
        // Redirect to login after 2 seconds
        setTimeout(() => {
            window.location.href = 'login.html';
        }, 2000);
    });
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.classList.add('show');
        
        setTimeout(() => {
            errorDiv.classList.remove('show');
        }, 5000);
    }
}

// Show success message
function showSuccess(message) {
    const successDiv = document.getElementById('successMessage');
    if (successDiv) {
        successDiv.textContent = message;
        successDiv.classList.add('show');
    }
}

// Logout function (to be called from main app)
function logout() {
    localStorage.removeItem('selaCurrentUser');
    window.location.href = 'login.html';
}

// Export logout function
window.logout = logout;
