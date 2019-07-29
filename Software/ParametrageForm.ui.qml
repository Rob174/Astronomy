import QtQuick 2.4

Item {
    width: 400
    height: 400
    property alias tuileForm3: tuileForm3
    property alias tuileForm: tuileForm

    Grid {
        id: grid
        spacing: 2
        rows: 2
        columns: 2
        anchors.fill: parent

        TuileForm {
            id: tuileForm
            width: (grid.height - grid.spacing) / 2
            height: (grid.height - grid.spacing) / 2
            state: "parametrage"
            text1Text: "Paramétrage"
        }

        TuileForm {
            id: tuileForm1
            width: (grid.height - grid.spacing) / 2
            height: (grid.height - grid.spacing) / 2
            state: "parametrage"
            text1Text: "Lieux"
        }

        TuileForm {
            id: tuileForm2
            width: (grid.height - grid.spacing) / 2
            height: (grid.height - grid.spacing) / 2
            state: "parametrage"
            text1Text: "Centres d'intérêt"
        }

        TuileForm {
            id: tuileForm3
            width: (grid.height - grid.spacing) / 2
            height: (grid.height - grid.spacing) / 2
            text1Text: "Paysages"
            state: "parametrage"
        }

    }
}

/*##^## Designer {
    D{i:1;anchors_height:400;anchors_width:400;anchors_x:36;anchors_y:35}
}
 ##^##*/
