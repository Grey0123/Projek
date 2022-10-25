
var doc = document.forms['submit']
var errorname = document.getElementById('error-name')
var erroremail = document.getElementById('error-email')
var errormessage = document.getElementById('error-message')

function validateform(e){
    // console.log(document.getElementById('fullname').value)
    
    if(doc['fullname'].value == ""){
        e.preventDefault();
        errorname.innerHTML = "<p> *Required <p>"
    }else{
        errorname.innerHTML = ""
    }
    if (doc['email'].value == ""){
        e.preventDefault();
        erroremail.innerHTML = "<p> *Required <p>"
    }else{
        errormail.innerHTML = ""
    }

    if(doc['message'].value == ""){
        e.preventDefault();
        errormessage.innerHTML = "<p> *Required <p>"
    }else{
        errormessage.innerHTML = ""
    }

}

