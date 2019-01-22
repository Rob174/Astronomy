import QtQuick 2.4

Item {
    id: item1
    signal clicked()
    property alias item1Height: item1.height
    property alias item1Width: item1.width
    property alias text1Text: text1.text
    property alias rectangle: rectangle
    property alias mouseArea: mouseArea

    MouseArea {
        id: mouseArea
        anchors.fill: parent
    }

    Rectangle {
        id: rectangle
        color: "#4b4b4b"
        border.color: "#00000000"
        anchors.fill: parent
    }

    Text {
        id: text1
        color: "#ffffff"
        text: qsTr("Text")
        fontSizeMode: Text.FixedSize
        anchors.rightMargin: 5
        anchors.leftMargin: 5
        anchors.bottomMargin: 5
        anchors.topMargin: 5
        wrapMode: Text.WordWrap
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        anchors.fill: parent
        font.pixelSize: parent.height / 8
    }

    Connections {
        target: mouseArea
        onClicked: clicked()
    }

    states: [
        State {
            name: "init"
        },
        State {
            name: "parametrage"

            PropertyChanges {
                target: rectangle
                color: "#61bd6d"
            }
        },
        State {
            name: "sortie"

            PropertyChanges {
                target: rectangle
                color: "#e14938"
            }
        },
        State {
            name: "plannification"

            PropertyChanges {
                target: rectangle
                color: "#54acd2"
            }
        },
        State {
            name: "postsortie"

            PropertyChanges {
                target: rectangle
                color: "#a38f84"
            }
        }
    ]
}

/*##^## Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
 ##^##*/

