import QtQuick 2.4

Item {
    id: accueil
    width: 400
    height: 400
    signal click(string url)
    Grid {
        id: grid
        anchors.fill: parent
        spacing: 2
        rows: 2
        columns: 2

        TuileForm {
            id: tuileForm
            item1Width: (grid.height - grid.spacing) / 2
            item1Height: (grid.height - grid.spacing) / 2
            text1Text: "Paramétrage"
            state: 'parametrage'
        }

        TuileForm {
            id: tuileForm1
            text1Text: "Sortie"
            item1Width: (grid.height - grid.spacing) / 2
            item1Height: (grid.height - grid.spacing) / 2
            state: 'sortie'
        }

        TuileForm {
            id: tuileForm2
            text1Text: "Plannification"
            item1Width: (grid.height - grid.spacing) / 2
            item1Height: (grid.height - grid.spacing) / 2
            state: 'plannification'
        }

        TuileForm {
            id: tuileForm3
            text1Text: "Post-sortie"
            item1Width: (grid.height - grid.spacing) / 2
            item1Height: (grid.height - grid.spacing) / 2
            state: 'postsortie'
        }
    }

    Connections {
        target: tuileForm
        onParentChanged: state = "parametrage"
    }

    Connections {
        target: tuileForm1
        onParentChanged: state = "sortie"
    }

    Connections {
        target: tuileForm2
        onParentChanged: state = "plannification"
    }

    Connections {
        target: tuileForm3
        onParentChanged: state = "postsortie"
    }

    Connections {
        target: tuileForm
        onClick: click('Parametrage.qml')
    }

    Connections {
        target: tuileForm1
        onClick: click('Sortie.qml')
    }

    Connections {
        target: tuileForm2
        onClick: click('Plannification.qml')
    }

    Connections {
        target: tuileForm3
        onClick: click('Postsortie.qml')
    }
}
