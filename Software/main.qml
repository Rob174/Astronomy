import QtQuick 2.11
import QtQuick.Window 2.11

Window {
    id: window
    visible: true
    width: 1920
    height: 1080
    title: qsTr("Astronomy_v2")
    //    visibility: Window.FullScreen
    //    flags: Qt.FramelessWindowHint
    color: "#4b4b4b"

    Rectangle {
        id: rectangle1
        color: "#444444"
        anchors.fill: parent
    }

    Rectangle {
        id: rectangle
        height: parent.height * 0.20
        color: "#4b4b4b"
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.left: parent.left
        anchors.leftMargin: 0
        anchors.top: parent.top
        anchors.topMargin: 0
    }

    Rectangle {
        id: banniere
        height: parent.height / 5
        color: "#00ffffff"
        anchors.leftMargin: rectangle.height / 4
        anchors.left: parent.left
        border.color: "#00000000"
        anchors.rightMargin: rectangle.height / 4
        anchors.right: parent.right
        anchors.bottomMargin: rectangle.height / 4
        anchors.bottom: rectangle.bottom
        anchors.topMargin: rectangle.height / 4
        anchors.top: rectangle.top
        Banniere {
            item: loader.children
        }
    }
    Loader {
        id: loader
        anchors.rightMargin: parent.height / 20
        anchors.leftMargin: parent.height / 20
        anchors.bottomMargin: parent.height / 20
        anchors.topMargin: parent.height / 20
        source: "Accueil.qml"
        anchors.top: rectangle.bottom
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.left: parent.left
    }

    Connections {
        target: loader.item
        onClicked: loader.setSource(url)
    }
}
