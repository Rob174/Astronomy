import QtQuick 2.0

Item {
    id: itemParent
    anchors.fill: parent
    property var item: []
    onItemChanged: {
        console.log('Modification')
        var array = []
        for (var i = 0; i < item.length; ++i) {
            console.log(item[i].type)
            var chaine = ''
            var a = 0
            while (a < item[i].type.length && item[i].type.slice(
                       a, a + 4) === 'Form' && item[i].type.slice(
                       a, a + 3) === '.ui') {
                chaine = item[i].type.slice(0, a + 1)
                a++
            }

            if (item.length - i - 1 == 0) {
                array.push([item[item.length - i - 1].type, 'f'])
            } else {
                array.push([item[item.length - i - 1].type, 'n'])
            }
        }
        repeater.model = array
        console.log('ModificationPost')
    }
    Row {
        anchors.fill: parent
        id: row
        spacing: 2
        layoutDirection: Qt.RightToLeft
        Repeater {
            anchors.fill: parent
            id: repeater
            model: ["patate", "Navet"]
            TuileHierarchie {
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                text1Text: modelData[0]
                state: {
                    if (modelData[1] === 'f') {
                        'StateFocus'
                    } else {
                        'StateNotFocus'
                    }
                }
            }
        }
    }
}
