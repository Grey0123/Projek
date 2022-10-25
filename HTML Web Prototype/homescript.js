$(function(){
    $("li#li").hide()
    var currImg = $("li#li").first()
    var currIdx = 0

    currImg.show()

    $("#next").click(function(){
        currImg.hide()

        if(currIdx != $("li#li").length - 1){
            currImg = currImg.next()
            currIdx += 1
        }else{
            currImg = $("li#li").first()
            currIdx = 0
        }
        currImg.show()
    })
    $("#prev").click(function(){
        currImg.hide()

        if(currIdx == 0){
            currImg = $("li#li").last()
            currIdx = $("li#li").length - 1
        }else{
            currImg = currImg.prev()
            currIdx -= 1
        }
        currImg.show()
    })
})
