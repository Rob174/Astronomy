import QtQuick 2.4

Item {
    signal click(string txt)
    id: tuile
    width: 400
    height: tuile.width / 3
    property alias text1Height: text1.height
    property alias text1Width: text1.width
    property alias text1Text: text1.text

    Rectangle {
        id: rectangle
        color: "#000000"
        opacity: 0.5
        anchors.fill: parent
    }

    MouseArea {
        id: mouseArea
        anchors.fill: parent
    }

    Text {
        id: text1
        color: "#ffffff"
        text: qsTr("Text")
        font.pixelSize: text1.height / 4
        wrapMode: Text.WrapAnywhere
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        fontSizeMode: Text.Fit
        anchors.fill: parent
    }

    Connections {
        target: mouseArea
        onClicked: click(text1.text)
    }
    states: [
        State {
            name: "StateFocus"
        },
        State {
            name: "StateNotFocus"

            PropertyChanges {
                target: rectangle
                opacity: 0.25
            }
        }
    ]
}

/*##^## Designer {
    D{i:1;anchors_height:200;anchors_width:200}D{i:5;anchors_height:100;anchors_width:100}
}
 ##^##*/
