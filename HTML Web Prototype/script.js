var doc = document.forms['submit']
var fullname = document.getElementById('error-name')
var beverage = document.getElementById('error-beverage')
var dessert = document.getElementById('error-dessert')
var city = document.getElementById('error-city')
var email = document.getElementById('error-email')
var address = document.getElementById('error-address')
var method = document.getElementById('error-method')
var card = document.getElementById('error-card')
var valid = document.getElementById('error-valid')
var cvv = document.getElementById('error-cvv')
var agree = document.getElementById('error-agree')
var total = document.getElementById('total')
var visamethod
document.getElementById('hide').style.display = "none";

function visadetail(e){
    document.getElementById('hide').style.display = "block";
    visamethod = true
}
function totalpay(e){
    
    var x = doc['menu-beverage'].value
    x = parseInt(x,10)
    var y = doc['menu-dessert'].value
    y = parseInt(y,10)
    if(x == -1){
        x = 0
    }
    if(y == -1){
        y = 0
    }
    var z = x + y
    total.innerText = "Rp" + z + ",00"
}

function paypaldetail(){
    document.getElementById('hide').style.display = "none";
    visamethod = false
}
function validateform(a){
    var temp = true
    if(doc['fullname'].value == ""){
        temp = false
        fullname.innerHTML = "<P>*Required Information<P>"
    }else{
        fullname.innerHTML = ""
    }

    if(doc['menu-beverage'].value == "-1"){
        temp = false
        beverage.innerHTML =  "<P>*Required Information<P>"
    }else{
        beverage.innerHTML = ""
    }

    if(doc['menu-dessert'].value == "-1"){
        temp = false
        dessert.innerHTML =  "<P>*Required Information<P>"
    }else{
        dessert.innerHTML = ""
    }

    if(doc['city'].value == ""){
        temp = false
        city.innerHTML =  "<P>*Required Information<P>"
    }else{
        city.innerHTML = ""
    }

    if(doc['email'].value == ""){
        temp = false
        email.innerHTML =  "<P>*Required Information<P>"
    }else{
        email.innerHTML = ""
    }

    if(doc['address'].value == ""){
        temp = false
        address.innerHTML =  "<P>*Required Information<P>"
    }else{
        address.innerHTML = ""
    }

    if(doc['method'].value == ""){
        temp = false
        method.innerHTML =  "<P>*Required Information<P>"
    }else{
        method.innerHTML = ""
    }
    if(visamethod == true){

        if(doc['card-number'].value == ""){
            temp = false
            card.innerHTML =  "<P>*Required Information<P>"
        }else{
            card.innerHTML = ""
        }

        if(doc['valid'].value == ""){
            temp = false
            valid.innerHTML =  "<P>*Required Information<P>"
        }else{
            valid.innerHTML = ""
        }
        if(doc['cvv'].value == ""){
            temp = false
            cvv.innerHTML =  "<P>*Required Information<P>"
        }else{
            cvv.innerHTML = ""
        }
    }


    if(doc['agree'].checked == false){
        temp = false
        agree.innerHTML =  "<P>*Required Information<P>"
    }else{
        agree.innerHTML = ""
    }


    if(temp == true){
    
        var confirmation = confirm('do you want to pay?')
        if(confirmation == true){
            alert('Thank you for your order')
        }

    }else{
        a.preventDefault()
    }
    // var x = doc['menu-beverage'].value
    // x = parseInt(x,10)
    // var y = doc['menu-dessert'].value
    // y = parseInt(y,10)
    // var z = x + y
    // total.innerText = "Rp" + z + ",00"
}
